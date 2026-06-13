use nalgebra::{Isometry3, Matrix3, Vector3};
use opencv::core::{KeyPoint, Mat};
use std::sync::{Arc, atomic::AtomicUsize};

use crate::{
    camera_models::GeometricCamera,
    imu_types::{Bias, Calib, Preintegrated},
    map_point::MapPoint,
    orb_vocabulary::{BowVector, FeatureVector},
};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

pub struct KeyFrame {
    pub imu: bool,

    pub id: u64,
    pub frame_id: usize,
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
    pub bias_gba: Bias,
    pub ba_global_for_kf: u64,

    // variables used by merging
    pub tcw_merge: Isometry3<f32>,
    pub tcw_bef_merge: Isometry3<f32>,
    pub twc_bef_merge: Isometry3<f32>,
    pub vwb_merge: Vector3<f32>,
    pub vwb_bef_merge: Vector3<f32>,
    pub bias_merge: Bias,
    pub merge_corrected_for_kf: u64,
    pub merge_for_kf: u64,
    pub scale_merge: f32,
    pub ba_local_for_merge: u64,

    pub scale: f64,

    // Calibration parameters
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub invfx: f32,
    pub invfy: f32,
    pub bf: f32,
    pub b: f32,
    pub th_depth: f32,
    pub dist_coef: Mat,

    // Number of KeyPoints
    pub n: u32,

    // Keypoints, stereo coordinate and descriptors (all associated by an index)
    pub keys: Vec<KeyPoint>,
    pub keys_un: Vec<KeyPoint>,
    // KeyPoints in the right image (for stereo fisheye, coordinates are needed).
    pub keys_right: Option<Vec<KeyPoint>>,
    pub u_right: Vec<f32>, // negative value for monocular points
    pub depth: Vec<f32>,   // negative value for monocular points
    pub descriptors: Mat,

    // BoW
    pub bow_vec: BowVector,
    pub feat_vec: FeatureVector,

    // Pose relative to parent (this is computed when bad flag is activated)
    pub tcp: Isometry3<f32>,

    // Scale
    pub scale_levels: usize,
    pub scale_factor: f32,
    pub log_scale_factor: f32,
    pub scale_factors: Vec<f32>,
    pub level_sigma2: Vec<f32>,
    pub inv_level_sigma2: Vec<f32>,

    // Image bounds and calibration
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,

    // Preintegrated IMU measurements from previous keyframes
    pub prev_kf: Arc<KeyFrame>,
    pub next_kf: Arc<KeyFrame>,

    pub imu_preintegrated: Arc<Preintegrated>,
    pub imu_calib: Calib,

    pub origin_map_id: u32,

    pub name_file: String,

    pub dataset: u32,

    pub loop_cand_kfs: Vec<Arc<KeyFrame>>,
    pub merge_cand_kfs: Vec<Arc<KeyFrame>>,

    pub camera: Arc<dyn GeometricCamera>,
    pub camera2: Option<Arc<dyn GeometricCamera>>,

    // Number of KeyPoints in the left and right images (stereo fisheye).
    pub n_left: Option<usize>,
    pub n_right: Option<usize>,

    // TODO: others here

    // The following variables originally needed to be accessed through a mutex to be thread safe
    // TODO: figure out how to approach this

    // Poses
    tcw: Isometry3<f32>,
    rcw: Matrix3<f32>,
    twc: Isometry3<f32>,
    rwc: Matrix3<f32>,

    // IMU position
    owb: Vector3<f32>,
    // Velocity (Only used for inertial SLAM)
    vw: Vector3<f32>,
    has_velocity: bool,

    // Transformation matrix between cameras in stereo fisheye
    tlr: Isometry3<f32>,
    trl: Isometry3<f32>,

    // IMU bias
    imu_bias: Bias,
    // MapPoints associated to keypoints
    //

    // TODO: HERE
}

impl KeyFrame {
    pub fn get_map_point_matches(&self) -> Vec<Option<Arc<MapPoint>>> {
        // TODO: signature might also change
        Vec::new()
    }

    pub fn is_in_image(&self, x: f32, y: f32) -> bool {
        // TODO
        true
    }

    pub fn get_features_in_area(&self, x: f32, y: f32, r: f32, right: bool) -> Vec<usize> {
        // TODO
        Vec::new()
    }

    pub fn get_pose(&self) -> &Isometry3<f32> {
        &self.tcw
    }
    pub fn get_pose_inverse(&self) -> &Isometry3<f32> {
        &self.twc
    }
    pub fn get_camera_center(&self) -> &Vector3<f32> {
        &self.twc.translation.vector
    }
    pub fn get_right_pose(&self) -> Isometry3<f32> {
        self.trl * self.tcw
    }
    pub fn get_right_pose_inverse(&self) -> Isometry3<f32> {
        self.twc * self.tlr
    }
}
