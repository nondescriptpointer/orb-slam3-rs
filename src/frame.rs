use nalgebra::{Isometry3, Matrix3, Vector3};
use opencv::core::Point2f;
use opencv::core::{KeyPoint, Mat};
use opencv::features2d::BFMatcher;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::camera_models::GeometricCamera;
use crate::imu_types::Bias;
use crate::imu_types::Calib;
use crate::imu_types::Preintegrated;
use crate::key_frame::KeyFrame;
use crate::map_point::MapPoint;
use crate::orb_extractor::OrbExtractor;
use crate::orb_vocabulary::BowVector;
use crate::orb_vocabulary::FeatureVector;
use crate::orb_vocabulary::OrbVocabulary;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
static FRAME_GRID_ROWS: usize = 48;
static FRAME_GRID_COLS: usize = 64;

pub struct Frame {
    // Current Frame id
    pub id: usize,

    // Vocabulary used for relocalization
    pub vocabulary: Arc<OrbVocabulary>,

    // Feature extractor. The right is used only in the stereo case.
    pub extractor_left: Arc<OrbExtractor>,
    pub extractor_right: Arc<OrbExtractor>,

    // Frame timestamp
    pub timestamp: f64,

    // Calibration matrix and OpenCV distortion parameters
    pub k: Mat,
    pub k_matrix: Matrix3<f32>,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub inv_fx: f32,
    pub inv_fy: f32,
    pub dist_coef: Mat,

    // Stereo baseline multiplied by fx
    pub b_fx: f32,

    // Stereo baseline in meters
    pub b: f32,

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points ar einserted as in the monocular case from 2 views.
    pub th_depth: f32,

    // Number of KeyPoints
    pub n: usize,

    // Vector of keypoints (originally for visualization) and undistorted (actually used by the system).
    // In the stereo case, keys_un is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    pub keys: Vec<KeyPoint>,
    pub keys_right: Vec<KeyPoint>,
    pub keys_un: Vec<KeyPoint>,

    // Corresponding stereo coordinate and depth for each keypoint
    pub map_points: Vec<Arc<MapPoint>>,
    // "Monocular" keypoints have a negative value
    pub u_right: Vec<f32>,
    pub depth: Vec<f32>,

    // Bag of Word Vector structures
    pub bow_vec: BowVector,
    pub feat_vec: FeatureVector,

    // ORB descriptor, each row associated to a keypoint
    pub descriptors: Mat,
    pub descriptors_right: Mat,

    // MapPoints associated to keypoints, None if no association
    // Flag to identify outlier associations.
    pub outlier: Option<Vec<bool>>,
    pub close_mps: usize,

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints
    pub grid_element_width_inv: f32,
    pub grid_element_height_inv: f32,
    pub grid: Vec<Vec<usize>>,

    // Prediction bias
    pub pred_bias: Bias,

    // IMU bias
    pub imu_bias: Bias,

    // IMU calibration
    pub imu_calib: Calib,

    // IMU preintegration from last keyframe
    pub imu_preintegrated: Arc<Preintegrated>,
    pub last_keyframe: Arc<KeyFrame>,

    // Pointer to previous frame
    pub prev_frame: Arc<Frame>,
    pub imu_preintegrated_frame: Arc<Preintegrated>,

    // Reference KeyFrame
    pub reference_kf: Arc<KeyFrame>,

    // Scale pyramid info
    pub scale_levels: usize,
    pub scale_factor: f32,
    pub log_scale_factor: f32,
    pub scale_factors: Vec<f32>,
    pub inv_scale_factors: Vec<f32>,
    pub level_sigma2: Vec<f32>,
    pub inv_level_sigma2: Vec<f32>,

    // Undistorted Image Bounds (computed once)
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,

    pub initial_computations: bool,

    pub project_ponts: HashMap<usize, Point2f>,
    pub matched_in_image: HashMap<usize, Point2f>,

    pub name_file: String,
    pub dataset: usize,

    pub camera: Arc<dyn GeometricCamera>,
    pub camera2: Arc<dyn GeometricCamera>,

    // Number of KeyPoints extracted in the left and right images
    pub n_left: usize,
    pub n_right: usize,
    // Number of non lapping KeyPoints
    pub mono_left: usize,
    pub mono_right: usize,

    // For stereo matching
    pub left_to_right_match: Vec<usize>,
    pub right_to_left_match: Vec<usize>,

    // For stereo fisheye matching
    pub bf_matcher: BFMatcher,

    // Triangulated stereo observations using as reference the left camera.
    // These are computed during compute_stereo_fish_eye_matches
    pub stereo_3d_points: Vec<Vector3<f32>>,

    // Grid for the right image
    pub grid_right: Vec<Vec<usize>>,

    #[cfg(feature = "register-times")]
    pub time_orb_ext: f64,
    #[cfg(feature = "register-times")]
    pub time_stereo_match: f64,

    // nalgebra migration
    t_cw: Isometry3<f32>,
    r_wc: Matrix3<f32>,
    o_w: Vector3<f32>,
    r_cw: Matrix3<f32>,
    t_cw_vec: Vector3<f32>,
    has_pose: bool,

    t_lr: Isometry3<f32>,
    t_rl: Isometry3<f32>,
    r_lr: Matrix3<f32>,
    t_lr_vec: Vector3<f32>,

    // IMU linear velocity
    vw: Vector3<f32>,
    has_velocity: bool,

    is_set: bool,
    is_imu_preintegrated: bool,
    // TODO? mutex
}

impl Frame {
    /*fn from_frame(frame: &Frame) -> Self {
        Frame { id: frame.id }
    }*/
}
