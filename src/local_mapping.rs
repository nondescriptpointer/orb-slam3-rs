use nalgebra::{Matrix3, SMatrix};
use std::sync::Arc;

use crate::{
    atlas::Atlas, key_frame::KeyFrame, loop_closing::LoopClosing, map::Map, map_point::MapPoint,
    system::System, tracking::Tracking,
};

pub struct LocalMapping {
    pub rwg: Matrix3<f64>,
    pub bg: Matrix3<f64>,
    pub ba: Matrix3<f64>,
    pub scale: f64,
    pub init_time: f64,
    pub cost_time: f64,
    pub init_sect: usize,
    pub idx_init: usize,
    pub n_kfs: usize,
    pub first_ts: f64,
    pub matches_inliers: usize,

    // For debugging (erase in normal mode)
    // TODO: evaluate
    pub init_fr: usize,
    pub idx_iteration: usize,
    pub sequence: String,

    pub not_ba1: bool,
    pub not_ba2: bool,
    pub bad_imu: bool,

    // not considered far points (clouds)
    pub far_points: bool,
    pub th_far_points: f32,

    #[cfg(feature = "register-times")]
    pub kf_insert_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub mp_culling_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub mp_creation_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub lba_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub kf_culling_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub lm_total_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub lba_sync_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub kf_culling_sync_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub lba_edges: Vec<usize>,
    #[cfg(feature = "register-times")]
    pub lba_kf_opt: Vec<usize>,
    #[cfg(feature = "register-times")]
    pub lba_kf_fixed: Vec<usize>,
    #[cfg(feature = "register-times")]
    pub lba_mps: Vec<usize>,
    #[cfg(feature = "register-times")]
    pub lba_exec: usize,
    #[cfg(feature = "register-times")]
    pub lba_abort: usize,

    system: Arc<System>,
    monocular: bool,
    inertial: bool,

    reset_requested: bool,
    reset_requested_active_map: bool,
    map_to_reset: Option<Arc<Map>>,

    finished_requested: bool,
    finished: bool,

    atlas: Arc<Atlas>,

    loop_closer: Option<Arc<LoopClosing>>,
    tracking: Option<Arc<Tracking>>,

    new_key_frames: Vec<Arc<KeyFrame>>,
    current_key_frame: Option<Arc<KeyFrame>>,

    recent_added_map_points: Vec<Arc<MapPoint>>,

    abort_ba: bool,

    stopped: bool,
    stop_requested: bool,
    not_stop: bool,

    accept_key_frames: bool,

    initializing: bool,

    info_inertial: SMatrix<f64, 9, 9>,
    num_lm: usize,
    num_kf_culling: usize,

    t_init: f32,
}

impl LocalMapping {
    fn new(
        system: Arc<System>,
        atlas: Arc<Atlas>,
        monocular: bool,
        inertial: bool,
        sequence: String,
    ) -> Self {
        LocalMapping {
            rwg: Matrix3::identity(),
            bg: Matrix3::identity(),
            ba: Matrix3::identity(),
            scale: 1.,
            init_time: 0.,
            cost_time: 0.,
            init_sect: 0,
            idx_init: 0,
            n_kfs: 0,
            first_ts: 0.,
            matches_inliers: 0,
            init_fr: 0,
            idx_iteration: 0,
            sequence,
            not_ba1: true,
            not_ba2: true,
            bad_imu: false,
            far_points: false, // forced off in the C++ code
            th_far_points: 0., // forced off in the C++ code
            #[cfg(feature = "register-times")]
            kf_insert_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            mp_culling_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            mp_creation_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            lba_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            kf_culling_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            lm_total_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            lba_sync_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            kf_culling_sync_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            lba_edges: Vec::new(),
            #[cfg(feature = "register-times")]
            lba_kf_opt: Vec::new(),
            #[cfg(feature = "register-times")]
            lba_kf_fixed: Vec::new(),
            #[cfg(feature = "register-times")]
            lba_mps: Vec::new(),
            #[cfg(feature = "register-times")]
            lba_exec: 0,
            #[cfg(feature = "register-times")]
            lba_abort: 0,
            system,
            monocular,
            inertial,
            reset_requested: false,
            reset_requested_active_map: false,
            map_to_reset: None,
            finished_requested: false,
            finished: true,
            atlas,
            loop_closer: None,
            tracking: None,
            new_key_frames: Vec::new(),
            current_key_frame: None,
            recent_added_map_points: Vec::new(),
            abort_ba: false,
            stopped: false,
            stop_requested: false,
            not_stop: false,
            accept_key_frames: true,
            initializing: false,
            info_inertial: SMatrix::zeros(),
            num_lm: 0,
            num_kf_culling: 0,
            t_init: 0.,
        }
    }
}
