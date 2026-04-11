use std::path::PathBuf;
use tracing::info;

use crate::settings::{Settings, SettingsError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sensor {
    Monocular = 0,
    Stereo = 1,
    RGBD = 2,
    IMUMonocular = 3,
    IMUStereo = 4,
    IMURGBD = 5,
}

pub struct System {
    sensor: Sensor,
    // TODO: viewer
    reset: bool,
    reset_active_map: bool,
    activate_localization_mode: bool,
    deactivate_localization_mode: bool,
    shutdown: bool,
}

#[derive(Debug)]
pub enum SystemError {
    InvalidSettings(SettingsError),
}

impl System {
    // TODO: handle initFr and strSequence
    pub fn new(
        vocabulary_path: &PathBuf,
        settings_path: &PathBuf,
        sensor: Sensor,
        use_viewer: bool,
    ) -> Result<Self, SystemError> {
        info!(
            "ORB-SLAM3 Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza."
        );
        info!(
            "ORB-SLAM2 Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza."
        );
        info!(
            "This program comes with ABSOLUTELY NO WARRANTY; This is free software, and you are welcome to redistribute it under certain conditions. See LICENSE."
        );
        info!("Input sensor was set to: {:?}", sensor);

        // Load settings
        let settings =
            Settings::new(settings_path, sensor).map_err(|e| SystemError::InvalidSettings(e))?;

        Ok(System {
            sensor,
            reset: false,
            reset_active_map: false,
            activate_localization_mode: false,
            deactivate_localization_mode: false,
            shutdown: false,
        })
    }
}
