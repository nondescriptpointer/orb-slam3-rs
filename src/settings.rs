use crate::system::Sensor;
use opencv::{
    core::{FileStorage, FileStorage_READ},
    prelude::*,
};
use std::path::PathBuf;
use tracing::info;

#[derive(Debug)]
pub enum CameraType {
    PinHole = 0,
    Rectified = 1,
    KannalaBrandt = 2,
}

#[derive(Debug)]
pub struct Settings {
    sensor: Sensor,
    camera_model: CameraType,
    need_to_undistort: bool,
    need_to_rectify: bool,
    need_to_resize1: bool,
    need_to_resize2: bool,
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
            panic!("Failed to open settings file");
        }
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
        let model =
            read_param_string(&settings, "Camera.type").ok_or(SettingsError::MissingCameraModel)?;
        let camera_model = match model.as_str() {
            "PinHole" => CameraType::PinHole,
            "Rectified" => CameraType::Rectified,
            "KannalaBrandt8" => CameraType::KannalaBrandt,
            _ => return Err(SettingsError::InvalidCameraModel(model)),
        };

        Ok(Settings {
            sensor,
            camera_model,
            need_to_undistort: false,
            need_to_rectify: false,
            need_to_resize1: false,
            need_to_resize2: false,
        })
    }

    fn read_camera(index: u8, storage: &FileStorage, model: &CameraType) {
        let index_one = index + 1;
        let calibration: Vec<f64>;
        match model {
            CameraType::PinHole => {
                // intrinsics
                let fx =
                    read_param_float(storage, &format!("Camera{}.fx", index_one)).unwrap_or(0.0);
                let fy =
                    read_param_float(storage, &format!("Camera{}.fy", index_one)).unwrap_or(0.0);
                let cx =
                    read_param_float(storage, &format!("Camera{}.cx", index_one)).unwrap_or(0.0);
                let cy =
                    read_param_float(storage, &format!("Camera{}.cy", index_one)).unwrap_or(0.0);
                calibration = vec![fx, fy, cx, cy];
            }
            CameraType::Rectified => {
                // TODO
            }
            CameraType::KannalaBrandt => {
                // TODO
            }
        }
    }
}

fn read_param_string(storage: &FileStorage, node: &str) -> Option<String> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_string().expect("expected string") {
            return Some(node.string().expect("expected string"));
        }
    }
    None
}

fn read_param_float(storage: &FileStorage, node: &str) -> Option<f64> {
    let node = storage.get(node);
    if let Ok(node) = node {
        if node.is_real().expect("expected float") {
            return Some(node.real().expect("expected float"));
        }
    }
    None
}
