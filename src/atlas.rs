use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, RwLock},
    time::Duration,
};
use tracing::info;

use crate::{
    camera_models::{GeometricCamera, Type},
    key_frame::KeyFrame,
    key_frame_database::KeyFrameDatabase,
    map::Map,
    map_point::MapPoint,
    orb_vocabulary::OrbVocabulary,
    viewer::Viewer,
};

struct Atlas {
    maps: HashMap<u32, Arc<RwLock<Map>>>,

    bad_maps: HashMap<u32, Arc<RwLock<Map>>>,

    current_map: Option<Arc<RwLock<Map>>>,

    cameras: Vec<Arc<dyn GeometricCamera>>,

    last_init_kf_id_map: u64,

    viewer: Option<Arc<Viewer>>,

    // Class references for the map reconstruction from the save field
    key_frame_database: Option<Arc<KeyFrameDatabase>>,
    orb_vocabulary: Option<Arc<OrbVocabulary>>,
}

impl Default for Atlas {
    fn default() -> Self {
        Self::new()
    }
}

impl Atlas {
    pub fn new() -> Self {
        Atlas {
            maps: HashMap::new(),
            bad_maps: HashMap::new(),
            current_map: None,
            cameras: Vec::new(),
            last_init_kf_id_map: 0,
            viewer: None,
            key_frame_database: None,
            orb_vocabulary: None,
        }
    }

    pub fn from_kf_id(id: u64) -> Self {
        let mut r = Atlas {
            last_init_kf_id_map: id,
            ..Atlas::new()
        };
        r.create_new_map();
        r
    }

    pub fn create_new_map(&mut self) {
        if let Some(current_map) = self.current_map.as_ref() {
            let max = current_map.read().unwrap().get_max_kf_id();
            if !self.maps.is_empty() && self.last_init_kf_id_map < max {
                self.last_init_kf_id_map = max + 1;
            }

            current_map.write().unwrap().set_stored_map();
            info!(
                "Stored map with ID: {}",
                current_map.read().unwrap().get_id()
            );
        }

        info!(
            "Creation of new map with last KF id: {}",
            self.last_init_kf_id_map
        );

        let new_map = Map::with_init_kf_id(self.last_init_kf_id_map);
        new_map.set_current_map();
        let id = new_map.get_id();
        info!("Creation of new map with ID: {}", id);
        let m = Arc::new(RwLock::new(new_map));
        self.current_map = Some(m.clone());
        self.maps.insert(id, m);
    }

    pub fn get_last_init_fk_id(&self) -> u64 {
        self.last_init_kf_id_map
    }

    pub fn set_viewer(&mut self, viewer: Arc<Viewer>) {
        self.viewer = Some(viewer);
    }

    // Method to change components in the current map
    pub fn add_key_frame(&mut self, kf: Arc<KeyFrame>) {
        if let Some(map) = kf.get_map() {
            map.add_key_frame(kf);
        }
    }

    pub fn add_map_point(&mut self, mp: Arc<MapPoint>) {
        if let Some(map) = mp.get_map() {
            map.add_map_point(mp);
        }
    }

    pub fn add_camera(&mut self, cam: Arc<dyn GeometricCamera>) -> Arc<dyn GeometricCamera> {
        // Check if the camera already exists
        let mut index_cam: Option<usize> = None;
        for (i, camera) in self.cameras.iter().enumerate() {
            if cam.is_equal(camera) {
                index_cam = Some(i);
            }
        }
        if let Some(index_cam) = index_cam {
            self.cameras[index_cam].clone()
        } else {
            self.cameras.push(cam.clone());
            cam
        }
    }

    pub fn get_all_cameras(&self) -> Vec<Arc<dyn GeometricCamera>> {
        self.cameras.clone()
    }

    // All methods without Map pointer work on current map
    pub fn set_reference_map_points(&self, mps: Vec<Arc<MapPoint>>) {
        if let Some(current_map) = self.current_map.as_ref() {
            current_map.write().unwrap().set_reference_map_points(mps);
        }
    }
    pub fn inform_new_big_change(&self) {
        if let Some(current_map) = self.current_map.as_ref() {
            current_map.write().unwrap().inform_new_big_change();
        }
    }
    pub fn get_last_big_change_idx(&self) -> Option<u32> {
        if let Some(current_map) = self.current_map.as_ref() {
            Some(current_map.read().unwrap().get_last_big_change_idx())
        } else {
            None
        }
    }

    pub fn map_points_in_map(&self) -> Option<usize> {
        if let Some(current_map) = self.current_map.as_ref() {
            Some(current_map.read().unwrap().map_points_in_map())
        } else {
            None
        }
    }
    pub fn key_frames_in_map(&self) -> Option<usize> {
        if let Some(current_map) = self.current_map.as_ref() {
            Some(current_map.read().unwrap().key_frames_in_map())
        } else {
            None
        }
    }

    // Method for get data in current map
    pub fn get_all_key_frames(&self) -> Option<Vec<Arc<KeyFrame>>> {
        if let Some(current_map) = self.current_map.as_ref() {
            Some(current_map.read().unwrap().get_all_key_frames())
        } else {
            None
        }
    }
    pub fn get_all_map_points(&self) -> Option<Vec<Arc<MapPoint>>> {
        if let Some(current_map) = self.current_map.as_ref() {
            Some(current_map.read().unwrap().get_all_map_points())
        } else {
            None
        }
    }
    pub fn get_reference_map_points(&self) -> Option<Vec<Arc<MapPoint>>> {
        if let Some(current_map) = self.current_map.as_ref() {
            Some(current_map.read().unwrap().get_reference_map_points())
        } else {
            None
        }
    }

    pub fn get_all_maps(&self) -> Vec<Arc<RwLock<Map>>> {
        let mut entries: Vec<_> = self.maps.iter().collect();
        entries.sort_by_key(|(id, _)| *id);
        entries
            .into_iter()
            .map(|(_, map)| Arc::clone(map))
            .collect()
    }

    pub fn count_maps(&self) -> usize {
        self.maps.iter().count()
    }

    pub fn clear_map(&mut self) {
        self.current_map.take();
    }

    pub fn clear_atlas(&mut self) {
        self.maps.clear();
        self.current_map.take();
        self.last_init_kf_id_map = 0;
    }

    // TODO: is this the right approach?
    pub fn get_current_map(&mut self) -> Arc<RwLock<Map>> {
        if let Some(current_map) = self.current_map.as_ref() {
            current_map.clone()
        } else {
            self.create_new_map();
            loop {
                if !self.current_map.as_ref().unwrap().read().unwrap().is_bad() {
                    break;
                }
                std::thread::sleep(Duration::from_micros(3000));
            }
            self.current_map.as_ref().unwrap().clone()
        }
    }

    pub fn set_map_bad(&mut self, map: Arc<RwLock<Map>>) {
        let id = map.read().unwrap().get_id();
        self.maps.remove(&id);
        map.write().unwrap().set_bad();
        self.bad_maps.insert(id, map);
    }

    pub fn remove_bad_maps(&mut self) {
        self.bad_maps.clear();
    }

    pub fn is_inertial(&self) -> Option<bool> {
        if let Some(current_map) = self.current_map.as_ref() {
            Some(current_map.read().unwrap().is_inertial())
        } else {
            None
        }
    }
    pub fn set_inertial_sensor(&mut self) {
        if let Some(current_map) = self.current_map.as_ref() {
            current_map.write().unwrap().set_inertial_sensor();
        }
    }
    pub fn set_imu_initialized(&mut self) {
        if let Some(current_map) = self.current_map.as_ref() {
            current_map.write().unwrap().set_imu_initialized();
        }
    }
    pub fn is_imu_initialized(&self) -> Option<bool> {
        if let Some(current_map) = self.current_map.as_ref() {
            Some(current_map.read().unwrap().is_imu_initialized())
        } else {
            None
        }
    }

    // Function for garantee the correction of serialization of this object
    pub fn pre_save(&mut self) {
        if let Some(current_map) = self.current_map.as_ref() {
            let max = current_map.read().unwrap().get_max_kf_id();
            if !self.maps.is_empty() && self.last_init_kf_id_map < max {
                self.last_init_kf_id_map = max + 1;
            }
        }

        let cams = self.cameras.clone();

        let maps_to_mark_bad: Vec<_> = self
            .maps
            .iter()
            .filter_map(|(_idx, map)| {
                {
                    let map_guard = map.read().unwrap();
                    if map_guard.is_bad() {
                        return None;
                    }
                    if map_guard.get_all_key_frames().is_empty() {
                        return Some(map.clone());
                    }
                }
                map.write().unwrap().pre_save(&cams);
                None
            })
            .collect();

        for map in maps_to_mark_bad {
            self.set_map_bad(map);
        }

        self.remove_bad_maps();
    }

    pub fn post_load(&mut self) {
        let cams: HashMap<u64, Arc<dyn GeometricCamera>> = self
            .cameras
            .iter()
            .map(|it| (it.get_id(), it.clone()))
            .collect();

        for (id, map) in &self.maps {
            map.write().unwrap().post_load(
                self.key_frame_database.as_ref().unwrap().clone(),
                self.orb_vocabulary.as_ref().unwrap().clone(),
            );
        }
    }

    pub fn get_atlas_key_frames(&self) -> HashMap<u32, Arc<KeyFrame>> {
        let mut ret = HashMap::new();
        for (id, map) in &self.maps {
            let keyframes = map.read().unwrap().get_all_key_frames();
            for kf in keyframes {
                ret.insert(*id, kf.clone());
            }
        }
        ret
    }

    pub fn set_key_frame_database(&mut self, db: Arc<KeyFrameDatabase>) {
        self.key_frame_database = Some(db);
    }
    pub fn get_key_frame_database(&self) -> Option<Arc<KeyFrameDatabase>> {
        self.key_frame_database.clone()
    }

    pub fn set_orb_vocabulary(&mut self, voc: Arc<OrbVocabulary>) {
        self.orb_vocabulary = Some(voc);
    }
    pub fn get_orb_vocabulary(&self) -> Option<Arc<OrbVocabulary>> {
        self.orb_vocabulary.clone()
    }

    pub fn get_num_lived_kdf(&self) -> usize {
        self.maps
            .values()
            .map(|map| map.read().unwrap().get_all_key_frames().len())
            .sum()
    }
    pub fn get_num_lived_mp(&self) -> usize {
        self.maps
            .values()
            .map(|map| map.read().unwrap().get_all_map_points().len())
            .sum()
    }
}
