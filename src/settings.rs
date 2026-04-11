use crate::{
    camera_models::{GeometricCamera, kannala_brandt8::KannalaBrandt8, pinhole::Pinhole},
    system::Sensor,
};
use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion};
use opencv::{
    core::{FileStorage, FileStorage_READ, Mat, Size},
    prelude::*,
};
use std::path::PathBuf;
use tracing::info;

#[derive(Debug, PartialEq, Eq)]
pub enum CameraType {
    PinHole = 0,
    Rectified = 1,
    KannalaBrandt = 2,
}

#[derive(Debug)]
struct CameraInfo {
    calibration: Box<dyn GeometricCamera>,
    original_calibration: Box<dyn GeometricCamera>,
    pinhole_distortion: Option<Vec<f32>>,
}

#[derive(Debug)]
struct ImageInfo {
    original_size: Size,
    new_size: Size,
    need_to_resize: bool,
    fps: i32,
    rgb: bool,
}

#[derive(Debug)]
struct IMUInfo {
    noise_gyro: f32,
    noise_acc: f32,
    gyro_walk: f32,
    gyro_acc: f32,
    imu_frequency: f32,
    tbc: Isometry3<f32>,
    insert_kfs_when_lost: bool,
}

#[derive(Debug)]
struct RGBDInfo {
    depth_map_factor: f32,
    th_depth: f32,
    b: f32,
    bf: f32,
}

#[derive(Debug)]
pub struct Settings {
    sensor: Sensor,
    camera_model: CameraType,
    camera1: Option<CameraInfo>,
    camera2: Option<CameraInfo>,
    image: ImageInfo,
    imu: Option<IMUInfo>,
    rgbd: Option<RGBDInfo>,
    need_to_rectify: bool,
    need_to_undistort: bool,
}

#[derive(Debug)]
pub enum SettingsError {
    InvalidFile,
    InvalidVersion,
    InvalidDatatype(String),
    MissingCameraModel,
    InvalidCameraModel(String),
}

impl Settings {
    pub fn new(config_file: &PathBuf, sensor: Sensor) -> Result<Self, SettingsError> {
        // Open settings file
        let settings = FileStorage::new_def(&config_file.to_string_lossy(), FileStorage_READ)
            .map_err(|_| SettingsError::InvalidFile)?;
        if let Err(_) = settings.is_opened() {
            return Err(SettingsError::InvalidFile);
        }
        info!("Load settings from {}", config_file.to_string_lossy());
        let node = settings
            .get("File.version")
            .map_err(|_| SettingsError::InvalidVersion)?;
        if !node
            .is_string()
            .map_err(|_| SettingsError::InvalidVersion)?
            || node.string().map_err(|_| SettingsError::InvalidVersion)? != "1.0"
        {
            return Err(SettingsError::InvalidVersion);
        }
        info!("Settings file is valid");

        // Get camera model
        let mut need_to_undistort = false;
        let model =
            read_param_string(&settings, "Camera.type").ok_or(SettingsError::MissingCameraModel)?;
        let camera_model = match model.as_str() {
            "PinHole" => CameraType::PinHole,
            "Rectified" => CameraType::Rectified,
            "KannalaBrandt8" => CameraType::KannalaBrandt,
            _ => return Err(SettingsError::InvalidCameraModel(model)),
        };

        // Get first camera
        let camera1 = Self::read_camera(0, &settings, &camera_model, &sensor);
        // Check if we need to correct distortion from the images
        if camera_model == CameraType::PinHole
            && matches!(sensor, Sensor::Monocular | Sensor::IMUMonocular)
            && camera1.pinhole_distortion.is_some()
        {
            need_to_undistort = true;
        }
        let mut camera1 = Some(camera1);

        // Get second camera if stereo (not rectified)
        let mut camera2 = None;
        let mut need_to_rectify = false;
        if matches!(sensor, Sensor::Stereo | Sensor::IMUStereo) {
            camera2 = Some(Self::read_camera(1, &settings, &camera_model, &sensor));
            if camera_model == CameraType::PinHole {
                need_to_rectify = true;
            }
        }

        // Image info
        let image = Self::read_image_info(
            &settings,
            need_to_rectify,
            &mut camera1,
            &mut camera2,
            &camera_model,
        );

        // IMU info
        let imu = if matches!(
            sensor,
            Sensor::IMUMonocular | Sensor::IMUStereo | Sensor::IMURGBD
        ) {
            Some(Self::read_imu_info(&settings))
        } else {
            None
        };

        // RGBD info
        let rgbd = if matches!(sensor, Sensor::RGBD | Sensor::IMURGBD) {
            Some(Self::read_rgbd_info(
                &settings,
                camera1
                    .as_ref()
                    .expect("missing camera")
                    .calibration
                    .get_parameter(0),
            ))
        } else {
            None
        };

        // TODO
        // ORB
        // Viewer
        // Load and save
        // Other params

        Ok(Settings {
            sensor,
            camera_model,
            camera1,
            camera2,
            need_to_rectify,
            need_to_undistort,
            image,
            imu,
            rgbd,
        })
    }

    fn read_camera(
        index: u8,
        storage: &FileStorage,
        model: &CameraType,
        sensor: &Sensor,
    ) -> CameraInfo {
        let index_one = index + 1;

        //Read intrinsic parameters
        let fx = read_param_float(storage, &format!("Camera{}.fx", index_one)).unwrap_or(0.0);
        let fy = read_param_float(storage, &format!("Camera{}.fy", index_one)).unwrap_or(0.0);
        let cx = read_param_float(storage, &format!("Camera{}.cx", index_one)).unwrap_or(0.0);
        let cy = read_param_float(storage, &format!("Camera{}.cy", index_one)).unwrap_or(0.0);

        match model {
            CameraType::PinHole => {
                let calibration = vec![fx, fy, cx, cy];
                let cam = Pinhole::with_params(calibration);
                let original = cam.clone();

                // Check if is a distorted PinHole
                let mut distortion = None;
                let k1 = read_param_float(storage, &format!("Camera{}.k1", index_one));
                if let Some(k1) = k1 {
                    let mut dist = vec![0.0f32; 4];
                    let k3 = read_param_float(storage, &format!("Camera{}.k3", index_one));
                    if let Some(k3) = k3 {
                        dist.push(k3);
                    }
                    dist[0] = k1;
                    dist[1] =
                        read_param_float(storage, &format!("Camera{}.k2", index_one)).unwrap_or(0.);
                    dist[2] =
                        read_param_float(storage, &format!("Camera{}.p1", index_one)).unwrap_or(0.);
                    dist[3] =
                        read_param_float(storage, &format!("Camera{}.p2", index_one)).unwrap_or(0.);
                    distortion = Some(dist);
                }

                CameraInfo {
                    calibration: Box::new(cam),
                    original_calibration: Box::new(original),
                    pinhole_distortion: distortion,
                }
            }
            CameraType::Rectified => {
                let calibration = vec![fx, fy, cx, cy];
                let cam = Pinhole::with_params(calibration);
                let original = cam.clone();

                // Rectified images are assumed to be ideal PinHole images (no distortion)
                CameraInfo {
                    calibration: Box::new(cam),
                    original_calibration: Box::new(original),
                    pinhole_distortion: None,
                }
            }
            CameraType::KannalaBrandt => {
                let k0 =
                    read_param_float(storage, &format!("Camera{}.k1", index_one)).unwrap_or(0.0);
                let k1 =
                    read_param_float(storage, &format!("Camera{}.k2", index_one)).unwrap_or(0.0);
                let k2 =
                    read_param_float(storage, &format!("Camera{}.k3", index_one)).unwrap_or(0.0);
                let k3 =
                    read_param_float(storage, &format!("Camera{}.k4", index_one)).unwrap_or(0.0);

                let calibration = vec![fx, fy, cx, cy, k0, k1, k2, k3];
                let mut cam = KannalaBrandt8::with_params(calibration);
                let original = cam.clone();

                // Overlapping
                if matches!(*sensor, Sensor::Stereo | Sensor::IMUStereo) {
                    let col_begin: usize =
                        read_param_int(storage, &format!("Camera{}.overlappingBegin", index_one))
                            .unwrap_or(0)
                            .try_into()
                            .unwrap_or(0);
                    let col_end: usize =
                        read_param_int(storage, &format!("Camera{}.overlappingEnd", index_one))
                            .unwrap_or(0)
                            .try_into()
                            .unwrap_or(0);
                    let overlapping = vec![col_begin, col_end];
                    cam.lapping_area = overlapping;
                }

                CameraInfo {
                    calibration: Box::new(cam),
                    original_calibration: Box::new(original),
                    pinhole_distortion: None,
                }
            }
        }
    }

    fn read_image_info(
        storage: &FileStorage,
        need_to_rectify: bool,
        camera1: &mut Option<CameraInfo>,
        camera2: &mut Option<CameraInfo>,
        camera_type: &CameraType,
    ) -> ImageInfo {
        // Read original and desired image dimensions
        let original_rows = read_param_int(storage, "Camera.height").unwrap_or(0);
        let original_cols = read_param_int(storage, "Camera.width").unwrap_or(0);
        let original_size = Size::new(
            original_cols.try_into().expect("size overflow"),
            original_rows.try_into().expect("size overflow"),
        );
        let mut new_size = original_size.clone();
        let mut need_to_resize = false;

        let new_height = read_param_int(storage, "Camera.newHeight");
        if let Some(new_height) = new_height {
            need_to_resize = true;
            new_size.height = new_height.try_into().expect("size oveflow");
            if !need_to_rectify {
                // Update calibration
                let scale_factor = new_size.height as f32 / original_size.height as f32;
                if let Some(camera) = camera1 {
                    camera
                        .calibration
                        .set_parameter(camera.calibration.get_parameter(1) * scale_factor, 1);
                    camera
                        .calibration
                        .set_parameter(camera.calibration.get_parameter(3) * scale_factor, 3);
                }
                if let Some(camera) = camera2 {
                    camera
                        .calibration
                        .set_parameter(camera.calibration.get_parameter(1) * scale_factor, 1);
                    camera
                        .calibration
                        .set_parameter(camera.calibration.get_parameter(3) * scale_factor, 3);
                }
            }
        }

        let new_width = read_param_int(storage, "Camera.newWidth");
        if let Some(new_width) = new_width {
            need_to_resize = true;
            new_size.width = new_width.try_into().expect("size overflow");
            if !need_to_rectify {
                // update calibration
                let scale_factor = new_size.width as f32 / original_size.width as f32;
                if let Some(camera) = camera1 {
                    camera
                        .calibration
                        .set_parameter(camera.calibration.get_parameter(0) * scale_factor, 0);
                    camera
                        .calibration
                        .set_parameter(camera.calibration.get_parameter(2) * scale_factor, 2);
                    if *camera_type == CameraType::KannalaBrandt {
                        if let Some(kanalla) = camera
                            .calibration
                            .as_any_mut()
                            .downcast_mut::<KannalaBrandt8>()
                        {
                            kanalla.lapping_area[0] =
                                (kanalla.lapping_area[0] as f32 * scale_factor) as usize;
                            kanalla.lapping_area[1] =
                                (kanalla.lapping_area[1] as f32 * scale_factor) as usize;
                        }
                    }
                }
                if let Some(camera) = camera2 {
                    camera
                        .calibration
                        .set_parameter(camera.calibration.get_parameter(0) * scale_factor, 0);
                    camera
                        .calibration
                        .set_parameter(camera.calibration.get_parameter(2) * scale_factor, 2);
                    if *camera_type == CameraType::KannalaBrandt {
                        if let Some(kanalla) = camera
                            .calibration
                            .as_any_mut()
                            .downcast_mut::<KannalaBrandt8>()
                        {
                            kanalla.lapping_area[0] =
                                (kanalla.lapping_area[0] as f32 * scale_factor) as usize;
                            kanalla.lapping_area[1] =
                                (kanalla.lapping_area[1] as f32 * scale_factor) as usize;
                        }
                    }
                }
            }
        }

        let fps = read_param_int(storage, "Camera.fps")
            .unwrap_or(30)
            .try_into()
            .unwrap_or(30);

        let rgb = read_param_bool(storage, "Camera.RGB").unwrap_or(false);

        ImageInfo {
            original_size,
            new_size,
            need_to_resize,
            fps,
            rgb,
        }
    }

    fn read_imu_info(storage: &FileStorage) -> IMUInfo {
        let cv_tbc = read_param_mat(storage, "IMU.T_b_c1");
        let mut tbc = Isometry3::identity();
        if let Some(cv_tbc) = cv_tbc {
            tbc = mat4x4_to_isometry3(&cv_tbc);
        }
        IMUInfo {
            noise_gyro: read_param_float(storage, "IMU.NoiseGyro").unwrap_or(0.),
            noise_acc: read_param_float(storage, "IMU.NoiseAcc").unwrap_or(0.),
            gyro_walk: read_param_float(storage, "IMU.GyroWalk").unwrap_or(0.),
            gyro_acc: read_param_float(storage, "IMU.AccWalk").unwrap_or(0.),
            imu_frequency: read_param_float(storage, "IMU.Frequency").unwrap_or(0.),
            tbc: tbc,
            insert_kfs_when_lost: read_param_bool(storage, "IMU.InsertKFsWhenLost").unwrap_or(true),
        }
    }

    fn read_rgbd_info(storage: &FileStorage, calib1_param: f32) -> RGBDInfo {
        let b = read_param_float(storage, "Stereo.b").unwrap_or(0.);
        RGBDInfo {
            depth_map_factor: read_param_float(storage, "RGBD.DepthMapFactor").unwrap_or(0.),
            th_depth: read_param_float(storage, "Stereo.ThDepth").unwrap_or(0.),
            b,
            bf: b * calib1_param,
        }
    }
}

fn read_param_string(storage: &FileStorage, node: &str) -> Option<String> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_string().expect("expected string") {
            return Some(node.to_string().expect("expected string"));
        }
    }
    None
}

fn read_param_float(storage: &FileStorage, node: &str) -> Option<f32> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_real().expect("expected float") {
            return Some(node.to_f32().expect("expected float"));
        }
    }
    None
}

fn read_param_double(storage: &FileStorage, node: &str) -> Option<f64> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_real().expect("expected double") {
            return Some(node.to_f64().expect("expected double"));
        }
    }
    None
}

fn read_param_int(storage: &FileStorage, node: &str) -> Option<i64> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_int().expect("expected float") {
            return Some(node.to_i64().expect("expected int"));
        }
    }
    None
}

fn read_param_bool(storage: &FileStorage, node: &str) -> Option<bool> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_int().expect("expected bool") {
            return Some(node.to_i32().expect("expected bool") == 1);
        }
    }
    None
}

fn read_param_mat(storage: &FileStorage, node: &str) -> Option<Mat> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if let Ok(mat) = node.mat() {
            return Some(mat);
        }
    }
    None
}

// Converts a 4×4 Mat (CV_64F row-major) into an Isometry3
// The matrix layout is:
// [ R  t ]
// [ 0  1 ]
// where R is 3×3 rotation and t is 3×1 translation.
fn mat4x4_to_isometry3(mat: &Mat) -> Isometry3<f32> {
    // YAML matrices are written as f64 (CV_64F) by OpenCV.
    let r = Matrix3::new(
        *mat.at_2d::<f64>(0, 0).expect("T_b_c1 row 0 col 0") as f32,
        *mat.at_2d::<f64>(0, 1).expect("T_b_c1 row 0 col 1") as f32,
        *mat.at_2d::<f64>(0, 2).expect("T_b_c1 row 0 col 2") as f32,
        *mat.at_2d::<f64>(1, 0).expect("T_b_c1 row 1 col 0") as f32,
        *mat.at_2d::<f64>(1, 1).expect("T_b_c1 row 1 col 1") as f32,
        *mat.at_2d::<f64>(1, 2).expect("T_b_c1 row 1 col 2") as f32,
        *mat.at_2d::<f64>(2, 0).expect("T_b_c1 row 2 col 0") as f32,
        *mat.at_2d::<f64>(2, 1).expect("T_b_c1 row 2 col 1") as f32,
        *mat.at_2d::<f64>(2, 2).expect("T_b_c1 row 2 col 2") as f32,
    );
    let t = Translation3::new(
        *mat.at_2d::<f64>(0, 3).expect("T_b_c1 row 0 col 3") as f32,
        *mat.at_2d::<f64>(1, 3).expect("T_b_c1 row 1 col 3") as f32,
        *mat.at_2d::<f64>(2, 3).expect("T_b_c1 row 2 col 3") as f32,
    );
    // from_matrix_unchecked is safe here: the source matrix is a rotation read
    // from a calibration file and is assumed to be orthonormal.
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    Isometry3::from_parts(t, rotation)
}
