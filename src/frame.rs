use nalgebra::{Isometry3, Matrix3, Vector3};
use opencv::calib3d::undistort_points;
use opencv::core::{CV_32F, KeyPoint, KeyPointTrait, KeyPointTraitConst, Mat, MatTraitConst};
use opencv::core::{MatTrait, Scalar};
use opencv::core::{NORM_HAMMING, Point2f};
use opencv::features2d::BFMatcher;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::camera_models::GeometricCamera;
use crate::converter::mat_to_matrix3f;
use crate::g2o_types::ConstraintPoseIMU;
use crate::imu_types::Bias;
use crate::imu_types::Calib;
use crate::imu_types::Preintegrated;
use crate::key_frame::KeyFrame;
use crate::map_point::MapPoint;
use crate::orb_extractor::{ExtractionError, OrbExtractResult, OrbExtractor};
use crate::orb_vocabulary::BowVector;
use crate::orb_vocabulary::FeatureVector;
use crate::orb_vocabulary::OrbVocabulary;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Number of rows/cols in the keypoint-to-cell grid
pub const FRAME_GRID_ROWS: usize = 48;
pub const FRAME_GRID_COLS: usize = 64;

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

    // Per-camera precomputed constants (intrinsics, bounds, grid sizing, K).
    // Replaces the C++ `Frame` statics guarded by `mbInitialComputations`.
    pub constants: Arc<FrameConstants>,

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
    pub prev_frame: Option<Arc<Frame>>,
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
    pub n_left: Option<usize>,
    pub n_right: Option<usize>,
    // Number of non lapping KeyPoints
    pub mono_left: Option<usize>,
    pub mono_right: Option<usize>,

    // For stereo matching
    pub left_to_right_match: Option<Vec<usize>>,
    pub right_to_left_match: Option<Vec<usize>>,

    // Triangulated stereo observations using as reference the left camera.
    // These are computed during compute_stereo_fish_eye_matches
    pub stereo_3d_points: Option<Vec<Vector3<f32>>>,

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
        constants: Arc<FrameConstants>,
        b_fx: f32,
        th_depth: f32,
        camera: Arc<dyn GeometricCamera>,
        prev_frame: Option<Arc<Frame>>,
        imu_calib: Calib,
    ) -> Self {
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
            let left_handle = s.spawn(|| extract_orb(extractor_left.as_ref(), im_left, 0, 0));
            let right_handle = s.spawn(|| extract_orb(extractor_right.as_ref(), im_right, 0, 0));
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
        let n = keys.len();

        // Undistort keypoints
        let camera_k = camera.to_k();
        let keys_un = undistort_keypoints(
            n as i32,
            &constants.dist_coef,
            &camera_k,
            &constants.k,
            &keys,
        );

        #[cfg(feature = "register-times")]
        let time_start_stareo_matches = std::time::Instant::now();

        // TODO: after completing stereo matching

        #[cfg(feature = "register-times")]
        let time_stereo_match = time_start_stareo_matches.elapsed().as_secs_f64() * 1000.0;

        let map_points = vec![None; n];
        let outlier = vec![false; n];
        let project_points = HashMap::new();
        let matched_in_image = HashMap::new();

        let b = b_fx / constants.intrinsics.fx;

        let mut has_velocity = false;
        let vw = if let Some(prev_frame) = prev_frame {
            if prev_frame.has_velocity {
                has_velocity = true;
                prev_frame.vw.clone()
            } else {
                Vector3::zeros()
            }
        } else {
            Vector3::zeros()
        };

        let (grid, grid_right) = assign_features_to_grid(
            n,
            None,
            &keys,
            Some(&keys_right),
            Some(&keys_un),
            &constants.bounds,
        );

        Frame {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            orb_vocabulary,
            extractor_left,
            extractor_right,
            timestamp,
            constants,
            b_fx,
            b,
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
            cpi: None,
            keys,
            keys_right: Some(keys_right),
            keys_un: Some(keys_un),
            descriptors,
            descriptors_right: Some(descriptors_right),
            n,
            map_points,
            outlier,
            project_points,
            matched_in_image,
            vw,
            has_velocity,
            // Set no stereo fisheye information
            n_left: None,
            n_right: None,
            mono_left: None,
            mono_right: None,
            left_to_right_match: None,
            right_to_left_match: None,
            stereo_3d_points: None,
            // features assigned to grid
            grid,
            grid_right,
            // TODO: here
        }
    }

    pub fn get_features_in_area(
        &self,
        x: f32,
        y: f32,
        r: f32,
        min_level: i32,
        max_level: i32,
        right: bool,
    ) -> Vec<usize> {
        // TODO
        Vec::new()
    }
}

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

/// Per-camera precomputed constants shared by every Frame produced from
/// the same calibration.
#[derive(Debug)]
pub struct FrameConstants {
    /// 3×3 intrinsics matrix as an OpenCV `Mat` (`CV_32F`).
    pub k: Mat,
    /// 3×3 intrinsics matrix as a nalgebra type.
    pub k_matrix: Matrix3<f32>,
    /// OpenCV distortion coefficients (may be empty / all-zero for fisheye).
    pub dist_coef: Mat,
    /// Scalar intrinsics derived from `k`.
    pub intrinsics: CameraIntrinsics,
    /// Undistorted image bounds and grid-cell inverse sizes.
    pub bounds: ImageBounds,
}

#[derive(Debug)]
pub enum FrameConstantsError {
    /// `k` could not be converted to a `Matrix3<f32>`.
    InvalidK(opencv::Error),
    /// Undistortion of the image corners failed.
    Undistort(opencv::Error),
    /// Image dimensions must be strictly positive.
    InvalidImageSize { cols: i32, rows: i32 },
}
impl std::fmt::Display for FrameConstantsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidK(e) => write!(f, "invalid camera matrix K: {e}"),
            Self::Undistort(e) => write!(f, "failed to undistort image corners: {e}"),
            Self::InvalidImageSize { cols, rows } => {
                write!(f, "invalid image size: {cols}x{rows}")
            }
        }
    }
}
impl std::error::Error for FrameConstantsError {}

impl FrameConstants {
    /// Build the per-camera constants from the calibration and the image size.
    ///
    /// This performs all the work the original C++ code did once inside the
    /// `if(mbInitialComputations)` block of every `Frame` constructor.
    ///
    /// `k` is the 3×3 intrinsics matrix (`CV_32F`). `dist_coef` is the OpenCV
    /// distortion-coefficient vector; pass an empty `Mat` (or one whose first
    /// element is zero) for cameras that don't use OpenCV-style distortion
    /// (e.g. fisheye/Kannala–Brandt), in which case the raw image rectangle is
    /// used as the bounds — matching the C++ behaviour.
    pub fn new(
        k: Mat,
        dist_coef: Mat,
        image_cols: i32,
        image_rows: i32,
    ) -> Result<Self, FrameConstantsError> {
        if image_cols <= 0 || image_rows <= 0 {
            return Err(FrameConstantsError::InvalidImageSize {
                cols: image_cols,
                rows: image_rows,
            });
        }

        let k_matrix = mat_to_matrix3f(&k).map_err(FrameConstantsError::InvalidK)?;
        let fx = k_matrix[(0, 0)];
        let fy = k_matrix[(1, 1)];
        let cx = k_matrix[(0, 2)];
        let cy = k_matrix[(1, 2)];
        let intrinsics = CameraIntrinsics {
            fx,
            fy,
            cx,
            cy,
            invfx: 1.0 / fx,
            invfy: 1.0 / fy,
        };

        let (min_x, max_x, min_y, max_y) =
            compute_image_bounds(&k, &dist_coef, image_cols, image_rows)?;
        let bounds = ImageBounds {
            min_x,
            max_x,
            min_y,
            max_y,
            grid_w_inv: FRAME_GRID_COLS as f32 / (max_x - min_x),
            grid_h_inv: FRAME_GRID_ROWS as f32 / (max_y - min_y),
        };

        Ok(Self {
            k,
            k_matrix,
            dist_coef,
            intrinsics,
            bounds,
        })
    }
}

fn compute_image_bounds(
    k: &Mat,
    dist_coef: &Mat,
    image_cols: i32,
    image_rows: i32,
) -> Result<(f32, f32, f32, f32), FrameConstantsError> {
    let has_distortion =
        !dist_coef.empty() && dist_coef.at::<f32>(0).map(|v| *v != 0.0).unwrap_or(false);

    if !has_distortion {
        return Ok((0.0, image_cols as f32, 0.0, image_rows as f32));
    }

    // 4×1 of CV_32FC2 — the format `undistortPoints` expects.
    let cols = image_cols as f32;
    let rows = image_rows as f32;
    let corners = Mat::from_slice_2d(&[
        [Point2f::new(0.0, 0.0)],
        [Point2f::new(cols, 0.0)],
        [Point2f::new(0.0, rows)],
        [Point2f::new(cols, rows)],
    ])
    .map_err(FrameConstantsError::Undistort)?;

    let mut undistorted = Mat::default();
    undistort_points(&corners, &mut undistorted, k, dist_coef, &Mat::default(), k)
        .map_err(FrameConstantsError::Undistort)?;

    let p = |i: i32| -> Result<Point2f, FrameConstantsError> {
        undistorted
            .at::<Point2f>(i)
            .copied()
            .map_err(FrameConstantsError::Undistort)
    };
    let (p0, p1, p2, p3) = (p(0)?, p(1)?, p(2)?, p(3)?);

    Ok((
        p0.x.min(p2.x),
        p1.x.max(p3.x),
        p0.y.min(p1.y),
        p2.y.max(p3.y),
    ))
}

fn bf_matcher() -> BFMatcher {
    BFMatcher::new(NORM_HAMMING, false).unwrap()
}

/// Run ORB extraction on a single image
fn extract_orb(
    extractor: &OrbExtractor,
    im: &Mat,
    x0: i32,
    x1: i32,
) -> Result<OrbExtractResult, ExtractionError> {
    let lapping = [x0, x1];
    extractor.compute(im, &Mat::default(), lapping)
}

pub fn pos_in_grid(kp: &KeyPoint, bounds: &ImageBounds) -> Option<(usize, usize)> {
    let pt = kp.pt();
    let pos_x = ((pt.x - bounds.min_x) * bounds.grid_w_inv).round() as i32;
    let pos_y = ((pt.y - bounds.min_y) * bounds.grid_h_inv).round() as i32;

    // Keypoints' coordinates are undistorted, which could push them outside
    // the image rectangle.
    if pos_x < 0 || pos_x >= FRAME_GRID_COLS as i32 || pos_y < 0 || pos_y >= FRAME_GRID_ROWS as i32
    {
        return None;
    }
    Some((pos_x as usize, pos_y as usize))
}
#[inline]
pub fn grid_index(col: usize, row: usize) -> usize {
    col * FRAME_GRID_ROWS + row
}
fn assign_features_to_grid(
    n: usize,
    n_left: Option<usize>,
    keys: &[KeyPoint],
    keys_right: Option<&[KeyPoint]>,
    keys_un: Option<&[KeyPoint]>,
    bounds: &ImageBounds,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n_cells = FRAME_GRID_COLS * FRAME_GRID_ROWS;
    let reserve = (0.5 * n as f32 / n_cells as f32) as usize;

    let mut grid: Vec<Vec<usize>> = (0..n_cells).map(|_| Vec::with_capacity(reserve)).collect();
    let mut grid_right: Vec<Vec<usize>> = if n_left.is_some() {
        (0..n_cells).map(|_| Vec::with_capacity(reserve)).collect()
    } else {
        Vec::new()
    };

    for i in 0..n {
        let kp = match n_left {
            None => &keys_un.expect("keys_un required for non-fisheye frames")[i],
            Some(nl) if i < nl => &keys[i],
            Some(nl) => &keys_right.expect("keys_right required for stereo-fisheye frames")[i - nl],
        };

        let Some((gx, gy)) = pos_in_grid(kp, bounds) else {
            continue;
        };
        let idx = grid_index(gx, gy);
        match n_left {
            None => grid[idx].push(i),
            Some(nl) if i < nl => grid[idx].push(i),
            Some(nl) => grid_right[idx].push(i - nl),
        }
    }

    (grid, grid_right)
}

fn undistort_keypoints(
    n: i32,
    dist_coef: &Mat,
    camera_k: &Mat,
    k: &Mat,
    keys: &[KeyPoint],
) -> Vec<KeyPoint> {
    if *dist_coef.at::<f32>(0).expect("get dist_coef") == 0. {
        return keys.to_vec();
    }

    // Fill matrix with points
    let mut mat =
        Mat::new_rows_cols_with_default(n, 2, CV_32F, Scalar::default()).expect("create mat");
    for (i, kp) in keys.iter().enumerate() {
        *mat.at_2d_mut::<f32>(i as i32, 0).unwrap() = kp.pt().x;
        *mat.at_2d_mut::<f32>(i as i32, 1).unwrap() = kp.pt().y;
    }

    // Reinterpret as Nx1 CV_32FC2, undistort, then back to Nx2 CV_32F.
    let mat_2c = mat.reshape(2, 0).expect("reshape to 2 channels");
    let mut undistorted = Mat::default();
    undistort_points(
        &mat_2c,
        &mut undistorted,
        camera_k,
        dist_coef,
        &Mat::default(),
        k,
    )
    .expect("undistortPoints");
    let undistorted = undistorted.reshape(1, 0).expect("reshape to 1 channel");

    // Write the undistorted coordinates back into a new keypoint vector.
    let mut keys_un = Vec::with_capacity(keys.len());
    for (i, kp) in keys.iter().enumerate() {
        let mut kp = kp.clone();
        let x = *undistorted.at_2d::<f32>(i as i32, 0).unwrap();
        let y = *undistorted.at_2d::<f32>(i as i32, 1).unwrap();
        kp.set_pt(Point2f::new(x, y));
        keys_un.push(kp);
    }
    keys_un
}

fn compute_stereo_matches(n: usize) {
    let u_right = vec![-1.0f32; n];
    let depth = vec![-1.0f32; n];

    let th_orb_dist = 0; // TODO
    // TODO
}
