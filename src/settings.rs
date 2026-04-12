use crate::{
    camera_models::{GeometricCamera, kannala_brandt8::KannalaBrandt8, pinhole::Pinhole},
    system::Sensor,
};
use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion};
use opencv::{
    calib3d::{CALIB_ZERO_DISPARITY, init_undistort_rectify_map, stereo_rectify},
    core::{CV_32F, CV_64F, FileStorage, FileStorage_READ, Mat, Rect, Size},
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

struct StereoInfo {
    tlr: Isometry3<f32>,
    b: f32,
    bf: f32,
    m1l: Mat,
    m2l: Mat,
    m1r: Mat,
    m2r: Mat,
}

// Mat does not implement Debug, so we hand-roll it.
impl std::fmt::Debug for StereoInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StereoInfo")
            .field("tlr", &self.tlr)
            .field("b", &self.b)
            .field("bf", &self.bf)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct ORBInfo {
    n_features: i32,
    scale_factor: f32,
    n_levels: i32,
    init_th_fast: i32,
    min_th_fast: i32,
}

#[derive(Debug)]
struct ViewerInfo {
    keyframe_size: f32,
    keyframe_linewidth: f32,
    graph_linewidth: f32,
    point_size: f32,
    camera_size: f32,
    camera_linewidth: f32,
    view_point_x: f32,
    view_point_y: f32,
    view_point_z: f32,
    view_point_f: f32,
    image_viewer_scale: f32,
}

#[derive(Debug)]
struct LoadAndSaveInfo {
    load_from: Option<String>,
    save_to: Option<String>,
}

#[derive(Debug)]
struct OtherInfo {
    th_far_points: Option<f32>,
}

#[derive(Debug)]
pub struct Settings {
    sensor: Sensor,
    camera_model: CameraType,
    camera1: Option<CameraInfo>,
    camera2: Option<CameraInfo>,
    need_to_rectify: bool,
    need_to_undistort: bool,
    image: ImageInfo,
    imu: Option<IMUInfo>,
    rgbd: Option<RGBDInfo>,
    stereo: Option<StereoInfo>,
    orb: ORBInfo,
    viewer: ViewerInfo,
    load_and_save: LoadAndSaveInfo,
    other: OtherInfo,
}

#[derive(Debug)]
pub enum SettingsError {
    InvalidFile,
    InvalidVersion,
    InvalidDatatype(String),
    MissingCameraModel,
    InvalidCameraModel(String),
    Rectification,
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
        let mut imu = if matches!(
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

        // Other info
        let orb = Self::read_orb(&settings);
        let viewer = Self::read_viewer(&settings);
        let load_and_save = Self::read_load_and_save(&settings);
        let other = Self::read_other(&settings);

        // Precompute rectification maps
        let mut stereo = None;
        if need_to_rectify {
            let k1 = camera1.as_ref().unwrap().calibration.to_k();
            let mut k1_converted = Mat::default();
            k1.convert_to(&mut k1_converted, CV_64F, 1., 0.)
                .map_err(|_| SettingsError::Rectification)?;
            let k2 = camera2.as_ref().unwrap().calibration.to_k();
            let mut k2_converted = Mat::default();
            k2.convert_to(&mut k2_converted, CV_64F, 1., 0.)
                .map_err(|_| SettingsError::Rectification)?;

            let tlr_mat =
                read_param_mat(&settings, "Stereo.T_c1_c2").ok_or(SettingsError::Rectification)?;
            let tlr = mat4x4_to_isometry3(&tlr_mat);

            // Invert Tlr_ to get the transform that maps points from cam1 to cam2
            // (R12, t12 as expected by stereoRectify).
            let tlr_inv = tlr.inverse();
            let r_mat = tlr_inv.rotation.to_rotation_matrix();
            let r = r_mat.matrix();
            let t = tlr_inv.translation.vector;
            let r12 = Mat::from_slice_2d(&[
                &[r[(0, 0)] as f64, r[(0, 1)] as f64, r[(0, 2)] as f64],
                &[r[(1, 0)] as f64, r[(1, 1)] as f64, r[(1, 2)] as f64],
                &[r[(2, 0)] as f64, r[(2, 1)] as f64, r[(2, 2)] as f64],
            ])
            .map_err(|_| SettingsError::Rectification)?;
            let t12 = Mat::from_slice_2d(&[&[t[0] as f64], &[t[1] as f64], &[t[2] as f64]])
                .map_err(|_| SettingsError::Rectification)?;

            let dist1_mat = match &camera1.as_ref().unwrap().pinhole_distortion {
                Some(d) => Mat::from_slice(d.as_slice())
                    .map_err(|_| SettingsError::Rectification)?
                    .try_clone()
                    .map_err(|_| SettingsError::Rectification)?,
                None => Mat::default(),
            };
            let dist2_mat = match &camera2.as_ref().unwrap().pinhole_distortion {
                Some(d) => Mat::from_slice(d.as_slice())
                    .map_err(|_| SettingsError::Rectification)?
                    .try_clone()
                    .map_err(|_| SettingsError::Rectification)?,
                None => Mat::default(),
            };

            let new_im_size = image.new_size;
            let mut r_r1_u1 = Mat::default();
            let mut r_r2_u2 = Mat::default();
            let mut p1 = Mat::default();
            let mut p2 = Mat::default();
            let mut q = Mat::default();
            let mut valid_roi1 = Rect::default();
            let mut valid_roi2 = Rect::default();
            stereo_rectify(
                &k1_converted,
                &dist1_mat,
                &k2_converted,
                &dist2_mat,
                new_im_size,
                &r12,
                &t12,
                &mut r_r1_u1,
                &mut r_r2_u2,
                &mut p1,
                &mut p2,
                &mut q,
                CALIB_ZERO_DISPARITY.into(),
                -1.,
                new_im_size,
                &mut valid_roi1,
                &mut valid_roi2,
            )
            .map_err(|_| SettingsError::Rectification)?;

            // P1 and P2 are 3×4; extract the 3×3 upper-left block as new camera matrices.
            let p1_3x3 =
                Mat::roi(&p1, Rect::new(0, 0, 3, 3)).map_err(|_| SettingsError::Rectification)?;
            let p2_3x3 =
                Mat::roi(&p2, Rect::new(0, 0, 3, 3)).map_err(|_| SettingsError::Rectification)?;

            let mut m1l = Mat::default();
            let mut m2l = Mat::default();
            init_undistort_rectify_map(
                &k1_converted,
                &dist1_mat,
                &r_r1_u1,
                &p1_3x3,
                new_im_size,
                CV_32F,
                &mut m1l,
                &mut m2l,
            )
            .map_err(|_| SettingsError::Rectification)?;

            let mut m1r = Mat::default();
            let mut m2r = Mat::default();
            init_undistort_rectify_map(
                &k2_converted,
                &dist2_mat,
                &r_r2_u2,
                &p2_3x3,
                new_im_size,
                CV_32F,
                &mut m1r,
                &mut m2r,
            )
            .map_err(|_| SettingsError::Rectification)?;

            // Update calibration
            let cam1 = camera1.as_mut().unwrap();
            cam1.calibration.set_parameter(
                *p1.at_2d::<f64>(0, 0)
                    .map_err(|_| SettingsError::Rectification)? as f32,
                0,
            );
            cam1.calibration.set_parameter(
                *p1.at_2d::<f64>(1, 1)
                    .map_err(|_| SettingsError::Rectification)? as f32,
                1,
            );
            cam1.calibration.set_parameter(
                *p1.at_2d::<f64>(0, 2)
                    .map_err(|_| SettingsError::Rectification)? as f32,
                2,
            );
            cam1.calibration.set_parameter(
                *p1.at_2d::<f64>(1, 2)
                    .map_err(|_| SettingsError::Rectification)? as f32,
                3,
            );

            // Update bf
            let b = read_param_float(&settings, "Stereo.b").unwrap_or(0.0);
            let bf = b
                * (*p1
                    .at_2d::<f64>(0, 0)
                    .map_err(|_| SettingsError::Rectification)? as f32);

            // Rectification rotates the left camera frame; propagate that into T_bc.
            if matches!(sensor, Sensor::IMUStereo) {
                if let Some(ref mut imu_info) = imu {
                    let get = |row: i32, col: i32| -> Result<f32, SettingsError> {
                        r_r1_u1
                            .at_2d::<f64>(row, col)
                            .map(|v| *v as f32)
                            .map_err(|_| SettingsError::Rectification)
                    };
                    let r_val = Matrix3::new(
                        get(0, 0)?,
                        get(0, 1)?,
                        get(0, 2)?,
                        get(1, 0)?,
                        get(1, 1)?,
                        get(1, 2)?,
                        get(2, 0)?,
                        get(2, 1)?,
                        get(2, 2)?,
                    );
                    // from_matrix_unchecked is safe: R_r1_u1 is an orthonormal rotation
                    // output from stereoRectify.
                    let rotation = UnitQuaternion::from_rotation_matrix(
                        &Rotation3::from_matrix_unchecked(r_val),
                    );
                    let t_r1_u1 = Isometry3::from_parts(Translation3::identity(), rotation);
                    imu_info.tbc = imu_info.tbc * t_r1_u1.inverse();
                }
            }

            stereo = Some(StereoInfo {
                tlr,
                b,
                bf,
                m1l,
                m2l,
                m1r,
                m2r,
            });
        }

        info!("Settings succesfully loaded");
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
            stereo,
            orb,
            viewer,
            load_and_save,
            other,
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
            noise_gyro: read_param_float(storage, "IMU.NoiseGyro").unwrap_or(1.7e-04),
            noise_acc: read_param_float(storage, "IMU.NoiseAcc").unwrap_or(2.0e-03),
            gyro_walk: read_param_float(storage, "IMU.GyroWalk").unwrap_or(1.9393e-05),
            gyro_acc: read_param_float(storage, "IMU.AccWalk").unwrap_or(3.0e-03),
            imu_frequency: read_param_float(storage, "IMU.Frequency").unwrap_or(200.0),
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

    fn read_orb(storage: &FileStorage) -> ORBInfo {
        ORBInfo {
            n_features: read_param_int(storage, "ORBextractor.nFeatures")
                .unwrap_or(1200)
                .try_into()
                .unwrap_or(1200),
            scale_factor: read_param_float(storage, "ORBextractor.scaleFactor").unwrap_or(0.),
            n_levels: read_param_int(storage, "ORBextractor.nLevels")
                .unwrap_or(8)
                .try_into()
                .unwrap_or(8),
            init_th_fast: read_param_int(storage, "ORBextractor.iniThFAST")
                .unwrap_or(20)
                .try_into()
                .unwrap_or(20),
            min_th_fast: read_param_int(storage, "ORBextractor.minThFAST")
                .unwrap_or(7)
                .try_into()
                .unwrap_or(7),
        }
    }

    fn read_viewer(storage: &FileStorage) -> ViewerInfo {
        ViewerInfo {
            keyframe_size: read_param_float(storage, "Viewer.KeyFrameSize").unwrap_or(0.05),
            keyframe_linewidth: read_param_float(storage, "Viewer.KeyFrameLineWidth")
                .unwrap_or(1.05),
            graph_linewidth: read_param_float(storage, "Viewer.GraphLineWidth").unwrap_or(0.9),
            point_size: read_param_float(storage, "Viewer.PointSize").unwrap_or(2.0),
            camera_size: read_param_float(storage, "Viewer.CameraSize").unwrap_or(0.08),
            camera_linewidth: read_param_float(storage, "Viewer.CameraLineWidth").unwrap_or(3.0),
            view_point_x: read_param_float(storage, "Viewer.ViewpointX").unwrap_or(0.0),
            view_point_y: read_param_float(storage, "Viewer.ViewpointY").unwrap_or(-0.7),
            view_point_z: read_param_float(storage, "Viewer.ViewpointZ").unwrap_or(-1.8),
            view_point_f: read_param_float(storage, "Viewer.ViewpointF").unwrap_or(500.0),
            image_viewer_scale: read_param_float(storage, "Viewer.imageViewScale").unwrap_or(1.0),
        }
    }

    fn read_load_and_save(storage: &FileStorage) -> LoadAndSaveInfo {
        LoadAndSaveInfo {
            load_from: read_param_string(storage, "System.LoadAtlasFromFile"),
            save_to: read_param_string(storage, "System.SaveAtlasToFile"),
        }
    }

    fn read_other(storage: &FileStorage) -> OtherInfo {
        OtherInfo {
            th_far_points: read_param_float(storage, "System.thFarPoints"),
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
