use std::path::PathBuf;
use tracing::info;

use crate::settings::{Settings, SettingsError};

#[derive(Debug,Clone,Copy)]
pub enum Sensor {
  Monocular=0,
  Stereo=1,
  RGBD=2,
  IMUMonocular=3,
  IMUStereo=4,
  IMURGBD=5,
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
  pub fn new(vocabulary_path: &PathBuf, settings_path: &PathBuf, sensor: Sensor, use_viewer: bool) -> Result<Self, SystemError> {
    info!("Input sensor was set to: {:?}", sensor);

    // Load settings
    let settings = Settings::new(settings_path, sensor).map_err(|e| SystemError::InvalidSettings(e))?;

    Ok(System {
      sensor,
      reset: false,
      reset_active_map: false,
      activate_localization_mode: false,
      deactivate_localization_mode: false,
      shutdown: false
    })
  }
}