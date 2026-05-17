use nalgebra::{Isometry3, Matrix3, Vector3};
use opencv::core::{KeyPoint, Mat, MatTraitConst};
use opencv::core::{NORM_HAMMING, Point2f};
use opencv::features2d::BFMatcher;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::camera_models::GeometricCamera;
use crate::converter::mat_to_matrix3f;
use crate::g2o_types::ConstraintPoseIMU;
use crate::imu_types::Bias;
use crate::imu_types::Calib;
use crate::imu_types::Preintegrated;
use crate::key_frame::KeyFrame;
use crate::map_point::MapPoint;
use crate::orb_extractor::{self, ExtractionError, OrbExtractResult, OrbExtractor};
use crate::orb_vocabulary::BowVector;
use crate::orb_vocabulary::FeatureVector;
use crate::orb_vocabulary::OrbVocabulary;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
static INITIAL_COMPUTATIONS: AtomicBool = AtomicBool::new(true);
static FRAME_GRID_ROWS: usize = 48;
static FRAME_GRID_COLS: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub invfx: f32,
    pub invfy: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ImageBounds {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub grid_w_inv: f32,
    pub grid_h_inv: f32,
}

#[derive(Clone)]
pub struct Frame {
    // Current Frame id
    pub id: usize,

    // Vocabulary used for relocalization
    pub orb_vocabulary: Arc<OrbVocabulary>,

    // Feature extractor. The right is used only in the stereo case.
    pub extractor_left: Arc<OrbExtractor>,
    pub extractor_right: Arc<OrbExtractor>,

    // Frame timestamp
    pub timestamp: f64,

    // Calibration matrix and OpenCV distortion parameters
    pub k: Mat,
    pub k_matrix: Matrix3<f32>,
    pub intrinsics: Arc<CameraIntrinsics>,
    pub bounds: Arc<ImageBounds>,
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
    pub keys_right: Option<Vec<KeyPoint>>,
    pub keys_un: Option<Vec<KeyPoint>>,

    // Corresponding stereo coordinate and depth for each keypoint
    pub map_points: Vec<Option<Arc<MapPoint>>>,
    // "Monocular" keypoints have a negative value
    pub u_right: Vec<f32>,
    pub depth: Vec<f32>,

    // Bag of Word Vector structures
    pub bow_vec: BowVector,
    pub feat_vec: FeatureVector,

    // ORB descriptor, each row associated to a keypoint
    pub descriptors: Mat,
    pub descriptors_right: Option<Mat>,

    // Flag to identify outlier associations.
    pub outlier: Vec<bool>,
    pub close_mps: usize,

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints
    pub grid: Vec<Vec<usize>>,

    // Prediction bias
    pub pred_bias: Bias,

    // IMU bias
    pub imu_bias: Bias,

    // IMU calibration
    pub imu_calib: Calib,

    // IMU preintegration from last keyframe
    pub imu_preintegrated: Option<Arc<Preintegrated>>,
    pub last_keyframe: Arc<KeyFrame>,

    // Pointer to previous frame
    pub prev_frame: Arc<Frame>,
    pub imu_preintegrated_frame: Option<Arc<Preintegrated>>,

    // Reference KeyFrame
    pub reference_kf: Option<Arc<KeyFrame>>,

    // Scale pyramid info
    pub scale_levels: usize,
    pub scale_factor: f32,
    pub log_scale_factor: f32,
    pub scale_factors: Vec<f32>,
    pub inv_scale_factors: Vec<f32>,
    pub level_sigma2: Vec<f32>,
    pub inv_level_sigma2: Vec<f32>,

    pub project_points: HashMap<usize, Point2f>,
    pub matched_in_image: HashMap<usize, Point2f>,

    pub name_file: String,
    pub dataset: usize,

    pub camera: Arc<dyn GeometricCamera>,
    pub camera2: Option<Arc<dyn GeometricCamera>>,

    // Number of KeyPoints extracted in the left and right images
    pub n_left: usize,
    pub n_right: Option<usize>,
    // Number of non lapping KeyPoints
    pub mono_left: usize,
    pub mono_right: Option<usize>,

    // For stereo matching
    pub left_to_right_match: Vec<usize>,
    pub right_to_left_match: Vec<usize>,

    // Triangulated stereo observations using as reference the left camera.
    // These are computed during compute_stereo_fish_eye_matches
    pub stereo_3d_points: Vec<Vector3<f32>>,

    // Grid for the right image
    pub grid_right: Vec<Vec<usize>>,

    #[cfg(feature = "register-times")]
    pub time_orb_ext: f64,
    #[cfg(feature = "register-times")]
    pub time_stereo_match: f64,

    cpi: Option<ConstraintPoseIMU>,

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
    fn from_stereo_cameras(
        im_left: &Mat,
        im_right: &Mat,
        timestamp: f64,
        extractor_left: Arc<OrbExtractor>,
        extractor_right: Arc<OrbExtractor>,
        orb_vocabulary: Arc<OrbVocabulary>,
        k: &Mat,
        dist_coef: &Mat,
        b_fx: f32,
        th_depth: f32,
        camera: Arc<dyn GeometricCamera>,
        prev_frame: Arc<Frame>,
        imu_calib: Calib,
    ) -> Self {
        let k_matrix = mat_to_matrix3f(k).expect("k matrix convert");
        let scale_levels = extractor_left.get_levels();
        let scale_factor = extractor_left.get_scale_factor();
        let log_scale_factor = scale_factor.ln();
        let scale_factors = extractor_left.get_scale_factors();
        let inv_scale_factors = extractor_left.get_inverse_scale_factors();
        let level_sigma2 = extractor_left.get_scale_sigma2();
        let inv_level_sigma2 = extractor_left.get_inverse_scale_sigma2();

        #[cfg(feature = "register-times")]
        let time_start_ext_orb = std::time::Instant::now();

        let (left_res, right_res) = std::thread::scope(|s| {
            let left_handle = s.spawn(|| Self::extract_orb(extractor_left.as_ref(), im_left, 0, 0));
            let right_handle =
                s.spawn(|| Self::extract_orb(extractor_right.as_ref(), im_right, 0, 0));
            let left = left_handle.join().expect("left ORB thread panicked");
            let right = right_handle.join().expect("right ORB thread panicked");
            (left, right)
        });
        let left_res = left_res.expect("left ORB extraction failed");
        let right_res = right_res.expect("right ORB extraction failed");

        #[cfg(feature = "register-times")]
        let time_orb_ext = time_start_ext_orb.elapsed().as_secs_f64() * 1000.0;

        let keys = left_res.keypoints;
        let keys_right = right_res.keypoints;
        let descriptors = left_res.descriptors.unwrap_or_default();
        let descriptors_right = right_res.descriptors.unwrap_or_default();
        let mono_left = left_res.mono_index.max(0) as usize;
        let mono_right = right_res.mono_index.max(0) as usize;
        let n_left = keys.len();
        let n_right = keys_right.len();
        let n = n_left;

        #[cfg(feature = "register-times")]
        let time_start_stareo_matches = std::time::Instant::now();

        // TODO: after completing stereo matching

        #[cfg(feature = "register-times")]
        let time_stereo_match = time_start_stareo_matches.elapsed().as_secs_f64() * 1000.0;

        let map_points = vec![None; n];
        let outlier = vec![false; n];
        let project_points = HashMap::new();
        let matched_in_image = HashMap::new();

        // This is done only for the first frame (or after a change in the calibration)
        if INITIAL_COMPUTATIONS.swap(false, Ordering::SeqCst) {
            // TODO: compute the image bounds
        }

        // TODO: mb = mbf / fx

        Frame {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            orb_vocabulary,
            extractor_left,
            extractor_right,
            timestamp,
            k: k.clone(),
            k_matrix,
            dist_coef: dist_coef.clone(),
            b_fx,
            th_depth,
            imu_calib,
            imu_preintegrated: None,
            prev_frame,
            imu_preintegrated_frame: None,
            reference_kf: None,
            is_set: false,
            is_imu_preintegrated: false,
            camera,
            camera2: None,
            has_pose: false,
            has_velocity: false,
            scale_levels,
            scale_factor,
            log_scale_factor,
            scale_factors,
            inv_scale_factors,
            level_sigma2,
            inv_level_sigma2,
            #[cfg(feature = "register-times")]
            time_orb_ext,
            #[cfg(feature = "register-times")]
            time_stereo_match,
            keys,
            keys_right: Some(keys_right),
            descriptors,
            descriptors_right: Some(descriptors_right),
            mono_left,
            mono_right: Some(mono_right),
            n_left,
            n_right: Some(n_right),
            n,
            map_points,
            outlier,
            project_points,
            matched_in_image,
            // TODO: here
            cpi: None,
        }
    }

    fn bf_matcher() -> BFMatcher {
        BFMatcher::new(NORM_HAMMING, false).unwrap()
    }

    /// Run ORB extraction on a single image. Analogous to the C++
    /// `Frame::ExtractORB(flag, im, x0, x1)` helper, except we return the
    /// `OrbExtractResult` instead of writing into `this`'s fields, so the
    /// caller can use it from a scoped thread without sharing `&mut Frame`.
    fn extract_orb(
        extractor: &OrbExtractor,
        im: &Mat,
        x0: i32,
        x1: i32,
    ) -> Result<OrbExtractResult, ExtractionError> {
        let lapping = [x0, x1];
        extractor.compute(im, &Mat::default(), lapping)
    }
}
