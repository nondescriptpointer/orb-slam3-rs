use nalgebra::Vector3;
use opencv::core::Mat;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    frame::Frame, key_frame::KeyFrame, key_frame_database::KeyFrameId, map::Map,
    serialization_utils,
};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Serialize, Deserialize, Default)]
pub struct MapPoint {
    pub id: usize,
    pub first_kf_id: Option<u64>,
    pub first_frame: usize,
    pub obs: i32,

    // Variables used by the tracking
    #[serde(skip)]
    pub track_proj_x: f32,
    #[serde(skip)]
    pub track_proj_y: f32,
    #[serde(skip)]
    pub track_proj_z: f32,
    #[serde(skip)]
    pub track_depth: f32,
    #[serde(skip)]
    pub track_depth_r: f32,
    #[serde(skip)]
    pub track_proj_xr: f32,
    #[serde(skip)]
    pub track_proj_yr: f32,
    #[serde(skip)]
    pub track_in_view: bool,
    #[serde(skip)]
    pub track_in_view_r: bool,
    #[serde(skip)]
    pub track_scale_level: i32,
    #[serde(skip)]
    pub track_scale_level_r: i32,
    #[serde(skip)]
    pub track_view_cos: f32,
    #[serde(skip)]
    pub track_view_cos_r: f32,
    #[serde(skip)]
    pub track_reference_for_frame: u64,
    #[serde(skip)]
    pub last_frame_seen: u64,

    // Variables used by local mapping
    #[serde(skip)]
    pub ba_local_for_kf: u64,
    #[serde(skip)]
    pub fuse_candidate_for_kf: u64,

    // Variables used for loop closing
    #[serde(skip)]
    pub loop_point_for_kf: u64,
    #[serde(skip)]
    pub corrected_by_kf: u64,
    #[serde(skip)]
    pub corrected_reference: u64,
    #[serde(skip)]
    pub pos_gba: Vector3<f32>,
    #[serde(skip)]
    pub ba_global_for_kf: u64,
    #[serde(skip)]
    pub ba_local_for_merge: u64,

    // Variables used by merging
    #[serde(skip)]
    pub pos_merge: Vector3<f32>,
    #[serde(skip)]
    pub normal_vector_merge: Vector3<f32>,

    // For inverse depth optimization
    #[serde(skip)]
    pub inv_depth: f64,
    #[serde(skip)]
    pub init_u: f64,
    #[serde(skip)]
    pub init_v: f64,
    #[serde(skip)]
    pub host_kf: Option<Arc<KeyFrame>>,

    #[serde(skip)]
    pub origin_map_id: u32,

    // Position in absolute coordinates
    world_pos: Vector3<f32>,

    // Keyframes observing the point and associated index in keyframe.
    // Rebuilt from backup_observations_id{1,2} after load.
    #[serde(skip)]
    observations: HashMap<KeyFrameId, (i32, i32)>,
    // For save relation without pointer, this is necessary for save/load function
    backup_observations_id1: HashMap<u64, i32>,
    backup_observations_id2: HashMap<u64, i32>,

    // Mean viewing direction
    normal_vector: Vector3<f32>,

    // Best descriptor to fast matching
    #[serde(with = "serialization_utils::mat_serde")]
    descriptor: Mat,

    // Reference KeyFrame — None until reconnected from backup_ref_kf_id.
    #[serde(skip)]
    ref_kf: Option<Arc<KeyFrame>>,
    backup_ref_kf_id: KeyFrameId,

    // Tracking counters
    #[serde(skip)]
    visible: i32,
    #[serde(skip)]
    found: i32,

    // Bad flag (we do not currently erase MapPoint from memory)
    bad: bool,
    // None until reconnected from backup_replace_id.
    #[serde(skip)]
    replaced: Option<Arc<MapPoint>>,
    // For save relation without pointer, this is necessary for save/load function
    backup_replace_id: u64,

    // Scale invariance distances
    min_distance: f32,
    max_distance: f32,

    #[serde(skip)]
    map: Option<Arc<Map>>,
}

impl MapPoint {
    pub fn new() -> Self {
        MapPoint {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            visible: 1,
            found: 1,
            bad: false,
            ..Default::default()
        }
    }
    pub fn from_pos_ref(pos: &Vector3<f32>, ref_kf: Arc<KeyFrame>, map: Arc<Map>) -> Self {
        MapPoint {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            first_kf_id: Some(ref_kf.id),
            first_frame: ref_kf.frame_id,
            ref_kf: Some(ref_kf),
            visible: 1,
            found: 1,
            bad: false,
            origin_map_id: map.get_id(),
            map: Some(map),
            world_pos: pos.clone(),
            normal_vector: Vector3::zeros(),
            track_in_view: false,
            track_in_view_r: false,
            ..Default::default()
        }
    }

    // TODO: here
    pub fn from_depth(pos: &Vector3<f32>, map: Arc<Map>, frame: &Frame, idx_f: i32) -> Self {
        // TODO: HERE this impl is not complete, depends on Frame
        MapPoint {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            first_frame: frame.id,
            visible: 1,
            found: 1,
            bad: false,
            origin_map_id: map.get_id(),
            map: Some(map),
            world_pos: pos.clone(),
            ..Default::default()
        }
    }

    pub fn is_bad(&self) -> bool {
        // TODO
        false
    }

    pub fn get_descriptor(&self) -> Mat {
        // TODO
        Mat::default()
    }

    pub fn observations(&self) -> i32 {
        // TODO
        0
    }

    pub fn get_world_pos(&self) -> Vector3<f32> {
        self.world_pos
    }

    // TODO: HERE
}
