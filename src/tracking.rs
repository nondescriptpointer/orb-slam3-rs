use crate::{
    atlas::Atlas,
    camera_models::GeometricCamera,
    frame::Frame,
    imu_types::{Bias, Calib, Point, Preintegrated},
    key_frame::KeyFrame,
    key_frame_database::KeyFrameDatabase,
    local_mapping::LocalMapping,
    loop_closing::LoopClosing,
    map_point::MapPoint,
    orb_extractor::OrbExtractor,
    orb_vocabulary::OrbVocabulary,
    settings::{CameraType, Settings},
    system::{Sensor, System},
};
use nalgebra::{Isometry3, Matrix3};
use opencv::core::{CV_32F, Mat, MatExprTraitConst, Point2f, Point3f, Scalar};
use opencv::prelude::*;
use std::{fs::File, fs::OpenOptions, io::BufWriter, sync::Arc};

pub struct Tracking {
    pub state: TrackingState,
    pub last_processed_state: TrackingState,

    // Input sensor
    pub sensor: Sensor,

    // Current frame
    pub current_frame: Frame,
    pub last_frame: Frame,

    pub im_gray: Mat,

    // Initialization variables (Monocular)
    pub ini_last_matches: Vec<usize>,
    pub ini_matches: Vec<usize>,
    pub prev_matched: Vec<Point2f>,
    pub ini_p3d: Vec<Point3f>,
    pub initial_frame: Frame,

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    pub relative_frame_poses: Vec<Isometry3<f32>>,
    pub references: Vec<Arc<KeyFrame>>,
    pub frame_times: Vec<f64>,
    pub lost: Vec<bool>,

    // Frames with estimated pose
    pub tracked_fr: usize,
    pub step: bool,

    // True if local mapping is deactivated and we are performing only localization
    pub only_tracking: bool,

    pub mean_track: f32,
    pub init_with_3kfs: bool,
    pub t0: f64,    // timestamp of first read frame
    pub t0vis: f64, // timestamp of first inserted keyframe
    pub t0imu: f64, // timestamp of IMU initialization
    pub fast_init: bool,

    pub write_stats: bool,

    #[cfg(feature = "register-times")]
    pub rect_stereo_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub resize_image_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub orb_extract_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub stereo_match_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub imu_integ_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub pose_pred_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub lm_track_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub new_kf_ms: Vec<f64>,
    #[cfg(feature = "register-times")]
    pub track_total_ms: Vec<f64>,

    pub im_right: Option<Mat>,

    map_updated: bool,

    // IMU preintegration from last frame
    imu_preintegrated_from_last_kf: Arc<Preintegrated>,

    // Queue of IMU measurements between frames
    queue_imu_data: Vec<Point>,

    // Vector of IMU measurements from previous to current frame (to be filled by preintegrate_imu)
    imu_from_last_frame: Vec<Point>,

    // IMU calibration parameters
    imu_calib: Vec<Arc<Calib>>,

    // Last Bias estimation (at keyframe creation)
    last_bias: Bias,

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    visual_odometry: bool,

    // Other thread pointers
    local_mapper: Arc<LocalMapping>,
    loop_closing: Arc<LoopClosing>,

    // ORB
    orb_extractor_left: Arc<OrbExtractor>,
    orb_extractor_right: Arc<OrbExtractor>,
    ini_orb_extractor: Arc<OrbExtractor>,

    // BoW
    orb_vocabulary: Arc<OrbVocabulary>,
    key_frame_database: Arc<KeyFrameDatabase>,

    // Initialization (only for monocular)
    ready_to_initialize: bool,
    set_init: bool,

    // Local map
    reference_kf: Arc<KeyFrame>,
    local_key_frames: Vec<Arc<KeyFrame>>,
    local_map_points: Vec<Arc<KeyFrame>>,

    // System
    system: Arc<System>,

    // Drawers
    // TODO: others
    #[cfg(feature = "viewer")]
    step_by_step: bool,

    // Atlas
    atlas: Arc<Atlas>,

    // Calibration matrix
    k: Mat,
    k_n: Matrix3<f32>,
    dist_coef: Mat,
    bf: f32,
    image_scale: f32,

    imu_freq: f32,
    imu_per: f64,
    insert_kfs_lost: bool,

    // New KeyFrame rules (according to fps)
    min_frames: u32,
    max_frames: u32,

    first_imu_frame_id: u32,
    frames_to_reset_imu: u64,

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points require a match in two keyframes
    th_depth: f32,

    // For RGB-D inputs only. For some datasets (eg TUM) the depthmap values are scaled
    depth_map_factor: f32,

    // Current matches in frame
    matches_inliers: usize,

    // Last Frame, KeyFrame and Relocalisation info
    last_key_frame: Option<Arc<KeyFrame>>,
    last_key_frame_id: usize,
    last_reloc_frame_id: usize,
    timestamp_lost: f64,
    timestamp_recently_lost: f64,

    first_frame_id: usize,
    initial_frame_id: usize,
    last_init_frame_id: usize,

    created_map: bool,

    // Motion model
    velocity: Option<Isometry3<f32>>,

    // Color order (true RGB, false BGR, ignored if grayscale)
    rgb: bool,

    temporal_points: Vec<Arc<MapPoint>>,

    num_dataset: usize,

    f_track_stats: BufWriter<File>,
    f_track_times: BufWriter<File>,

    time_pre_int_imu: f64,
    time_pose_pred: f64,
    time_local_map_track: f64,
    time_new_kf_dec: f64,

    camera: Arc<dyn GeometricCamera>,
    camera2: Option<Arc<dyn GeometricCamera>>,

    init_id: usize,
    last_id: usize,

    #[cfg(feature = "tracker-pause-resume")]
    stopped: bool,
    stop_requested: bool,
    not_stop: bool,

    tlr: Option<Isometry3<f32>>,
}

pub enum TrackingState {
    SystemNotReady,
    NoImagesYet,
    NotInitialized,
    OK,
    RecentlyLost,
    Lost,
    OKKLT,
}

impl Tracking {
    fn new(
        system: Arc<System>,
        orb_vocabulary: Arc<OrbVocabulary>,
        atlas: Arc<Atlas>,
        key_frame_database: Arc<KeyFrameDatabase>,
        settings_path: String,
        sensor: Sensor,
        settings: Arc<Settings>,
    ) -> Self {
        // Load camera parameters from settings file
        let camera = settings
            .camera1
            .as_ref()
            .expect("settings missing camera1")
            .calibration
            .clone();
        let camera = atlas.add_camera(camera);

        let dist_coef = if settings.need_to_undistort {
            settings.camera1_distortion_coef()
        } else {
            Mat::new_rows_cols_with_default(4, 1, CV_32F, Scalar::all(0.0)).unwrap()
        };

        let fx = camera.get_parameter(0);
        let fy = camera.get_parameter(1);
        let cx = camera.get_parameter(2);
        let cy = camera.get_parameter(3);
        let k = Mat::eye(3, 3, CV_32F).unwrap().to_mat().unwrap();
        *k.at_2d_mut::<f32>(0, 0).unwrap() = fx;
        *k.at_2d_mut::<f32>(1, 1).unwrap() = fy;
        *k.at_2d_mut::<f32>(0, 2).unwrap() = cx;
        *k.at_2d_mut::<f32>(1, 2).unwrap() = cy;
        let k_n: Matrix3<f32> = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

        let (camera2, tlr) =
            if matches!(sensor, Sensor::Stereo | Sensor::IMUStereo | Sensor::IMURGBD)
                && settings.camera_model == CameraType::KannalaBrandt
            {
                let ret = settings.camera2.as_ref().unwrap().calibration.clone();
                let ret = atlas.add_camera(ret);
                (
                    Some(ret),
                    Some(settings.stereo.expect("missing stereo info").tlr),
                )
            } else {
                (None, None)
            };

        Tracking {
            state: TrackingState::NoImagesYet,
            sensor,
            tracked_fr: 0,
            step: false,
            only_tracking: false,
            map_updated: false,
            visual_odometry: false,
            orb_vocabulary,
            key_frame_database,
            ready_to_initialize: false,
            system,
            #[cfg(feature = "viewer")]
            step_by_step: false,
            atlas,
            last_reloc_frame_id: 0,
            timestamp_recently_lost: 5.,
            initial_frame_id: 0,
            created_map: false,
            first_frame_id: 0,
            last_key_frame: None,
            init_id: 0,
            last_id: 0,
            init_with_3kfs: false,
            num_dataset: 0,
            camera,
            camera2,
            tlr,
            dist_coef,
            image_scale: 1.,
            k,
            k_n,
            #[cfg(feature = "register-times")]
            rect_stereo_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            resize_image_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            orb_extract_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            stereo_match_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            imu_integ_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            pose_pred_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            lm_track_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            new_kf_ms: Vec::new(),
            #[cfg(feature = "register-times")]
            track_total_ms: Vec::new(),
        }
    }

    #[cfg(feature = "register-times")]
    pub fn local_map_stats_to_file(&self) {
        // TODO
    }
    #[cfg(feature = "register-times")]
    pub fn track_stats_to_file(&self) {
        // TODO
    }
    #[cfg(feature = "register-times")]
    pub fn print_time_stats(&self) {
        // TODO
    }

    #[cfg(feature = "tracker-pause-resume")]
    pub fn request_stop(&self) {
        // TODO
    }
    #[cfg(feature = "tracker-pause-resume")]
    pub fn is_stopped(&self) -> bool {
        // TODO
        false
    }
    #[cfg(feature = "tracker-pause-resume")]
    pub fn release(&self) {
        // TODO
    }
    #[cfg(feature = "tracker-pause-resume")]
    pub fn stop_requested(&self) {
        // TODO
    }
    #[cfg(feature = "tracker-pause-resume")]
    fn stop(&self) {
        // TODO
    }
}

fn load_settings(settings: Arc<Settings>) {}
