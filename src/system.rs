use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::{
    atlas::Atlas,
    key_frame_database::KeyFrameDatabase,
    orb_vocabulary::{OrbVocabulary, VocabularyError},
    settings::{Settings, SettingsError},
    viewer::Viewer,
};

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
    // TODO
    viewer: Option<Arc<Viewer>>,
    reset: bool,
    reset_active_map: bool,
    activate_localization_mode: bool,
    deactivate_localization_mode: bool,
    shutdown: bool,
    vocabulary_file_path: PathBuf,
    load_atlas_file_path: Option<String>,
    save_atlas_file_path: Option<String>,
}

#[derive(Debug)]
pub enum SystemError {
    InvalidSettings(SettingsError),
    InvalidORBVocabulary(VocabularyError),
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
        let loop_closing = settings.loop_closing;

        // Load ORB VOB vocabulary
        info!("Loading ORB vocabulary. This could take a while...");
        let vocabulary = OrbVocabulary::load_from_text_file(vocabulary_path)
            .map_err(|e| SystemError::InvalidORBVocabulary(e))?;
        info!("ORB vocabulary loaded!");
        let vocabulary = Arc::new(vocabulary);
        // Create keyframe database
        let keyframe_database = KeyFrameDatabase::new(vocabulary);

        let atlas = if let Some(load_path) = &settings.load_and_save.load_from {
            info!("Initialization of Atlas from file: {}", load_path);
            // TODO: here
            Atlas::new()
        } else {
            info!("Initialization of Atlas from scratch");
            Atlas::from_kf_id(0)
        };

        // TODO: here

        Ok(System {
            sensor,
            viewer: None,
            reset: false,
            reset_active_map: false,
            activate_localization_mode: false,
            deactivate_localization_mode: false,
            shutdown: false,
            vocabulary_file_path: vocabulary_path.clone(),
            load_atlas_file_path: settings.load_and_save.load_from.clone(),
            save_atlas_file_path: settings.load_and_save.save_to.clone(),
        })
    }
}

fn load_atlas() -> Atlas {
    Atlas::new()
}
