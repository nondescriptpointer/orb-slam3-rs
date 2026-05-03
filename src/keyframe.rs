use nalgebra::{Isometry3, Vector3};

pub struct Keyframe {
    imu: bool,
    pub next_id: u64,
    pub id: u64,
    pub frame_id: u64,
    pub timestamp: f64,
    // Grid (to speed up feature matching)
    pub grid_cols: u32,
    pub grid_rows: u32,
    pub grid_element_width_inv: f32,
    pub grid_element_height_inv: f32,
    // Variables used by the tracking
    pub track_reference_for_frame: u64,
    pub fuse_target_for_kf: u64,
    // Variables used by local mapping
    pub ba_local_for_kf: u64,
    pub ba_fixed_for_kf: u64,
    // Number of optimizations by BA (amount of iterations in BA)
    pub number_of_opt: u64,
    // Variables used by the keyframe database
    pub loop_query: u64,
    pub loop_words: i32,
    pub loop_score: f32,
    pub reloc_query: u64,
    pub reloc_words: i32,
    pub reloc_score: f32,
    pub merge_query: u64,
    pub merge_words: i32,
    pub merge_score: f32,
    pub place_recognition_query: u64,
    pub place_recognition_words: i32,
    pub place_recognition_score: f32,
    pub current_place_recognition: bool,
    // Variables used by loop closing
    pub tcw_gba: Isometry3<f32>,
    pub tcw_bef_gba: Isometry3<f32>,
    pub vwb_gba: Vector3<f32>,
    pub vwb_bef_gba: Vector3<f32>,
    // TODO
}

impl Keyframe {}
