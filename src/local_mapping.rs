use nalgebra::{DMatrix, Matrix3};
use std::sync::Arc;

use crate::{
    atlas::Atlas, key_frame::KeyFrame, loop_closing::LoopClosing, map::Map, map_point::MapPoint,
    system::System, tracking::Tracking,
};

pub struct LocalMapping {
    pub cov_inertial: DMatrix<f32>,
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

    pub write_stats: bool,

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
    map_to_reset: Arc<Map>,

    finished_requested: bool,
    finished: bool,

    atlas: Arc<Atlas>,

    loop_closer: Arc<LoopClosing>,
    tracking: Arc<Tracking>,

    new_key_frames: Vec<Arc<KeyFrame>>,
    current_key_frame: Arc<KeyFrame>,

    recent_added_map_points: Vec<Arc<MapPoint>>,

    abort_ba: bool,

    stopped: bool,
    stop_requested: bool,
    not_stop: bool,

    accept_key_frames: bool,

    initializing: bool,

    info_inertial: DMatrix<f64>,
    num_lm: usize,
    num_kf_culling: usize,

    t_init: f32,
    count_refinement: usize,
}
