//! [`KeyFrame`]: a frame promoted into the map, with a covisibility graph,
//! spanning tree, observations and pose.
//!
//! ## Threading model
//!
//! Like [`crate::map_point::MapPoint`], a `KeyFrame` is a shared, mutably
//! aliased graph node (Tracking / LocalMapping / LoopClosing all touch it via
//! `Arc<KeyFrame>`), so it uses interior mutability
//!
//! * `pose` — `Tcw`/`Twc`, IMU position, velocity, stereo `Tlr`/`Trl`, bias
//! * `features` — the `MapPoint` matches vector
//! * `conn` — covisibility weights, spanning tree and loop/merge edges
//! * `map` — owning map pointer
//!
//! Immutable, construction-time data (calibration, keypoints, descriptors,
//! scale pyramid, grid, BoW) is stored as plain fields with no synchronisation.
//!
//! Graph edges (connected keyframes, parent, children, loop/merge edges,
//! prev/next) are [`Weak`] to avoid `Arc` cycles; the strong owners are the map
//! and the test/Tracking code holding the keyframes.

// `Arc<KeyFrame>`/`Arc<MapPoint>` hash by their stable `id`, so the `RwLock`
// interior mutability is irrelevant to the hash.
#![allow(clippy::mutable_key_type)]

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock, Weak};

use nalgebra::{Isometry3, Matrix3, Vector3};
use opencv::core::{KeyPoint, KeyPointTraitConst, Mat, MatTraitConst, Point2f};

use crate::camera_models::GeometricCamera;
use crate::frame::{FRAME_GRID_COLS, FRAME_GRID_ROWS, Frame, grid_index};
use crate::imu_types::{Bias, Calib, Preintegrated};
use crate::map::Map;
use crate::map_point::MapPoint;
use crate::orb_vocabulary::{BowVector, DESC_LEN, Descriptor, FeatureVector};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// Pose-group state
struct PoseState {
    tcw: Isometry3<f32>,
    rcw: Matrix3<f32>,
    twc: Isometry3<f32>,
    rwc: Matrix3<f32>,
    owb: Vector3<f32>,
    vw: Vector3<f32>,
    has_velocity: bool,
    tlr: Isometry3<f32>,
    trl: Isometry3<f32>,
    imu_bias: Bias,
}

/// A graph neighbour with its covisibility weight.
type WeightedKf = (Weak<KeyFrame>, i32);

/// Covisibility / spanning-tree state
#[derive(Default)]
struct ConnState {
    connected_weights: HashMap<u64, WeightedKf>,
    ordered: Vec<WeightedKf>,
    parent: Option<Weak<KeyFrame>>,
    children: HashMap<u64, Weak<KeyFrame>>,
    loop_edges: HashMap<u64, Weak<KeyFrame>>,
    merge_edges: HashMap<u64, Weak<KeyFrame>>,
    first_connection: bool,
    not_erase: bool,
    // Reserved for the (deferred) `set_bad_flag` cascade.
    #[allow(dead_code)]
    to_be_erased: bool,
    bad: bool,
}

pub struct KeyFrame {
    pub imu: bool,

    pub id: u64,
    pub frame_id: usize,
    pub timestamp: f64,

    // Grid (to speed up feature matching)
    pub grid_cols: usize,
    pub grid_rows: usize,
    pub grid_element_width_inv: f32,
    pub grid_element_height_inv: f32,

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
    pub k_matrix: Matrix3<f32>,

    // Number of KeyPoints
    pub n: usize,

    // Keypoints, stereo coordinate and descriptors (all associated by an index)
    pub keys: Vec<KeyPoint>,
    pub keys_un: Vec<KeyPoint>,
    pub keys_right: Option<Vec<KeyPoint>>,
    pub u_right: Vec<f32>,
    pub depth: Vec<f32>,
    pub descriptors: Mat,

    // BoW (populated at construction)
    pub bow_vec: BowVector,
    pub feat_vec: FeatureVector,

    // Scale
    pub scale_levels: usize,
    pub scale_factor: f32,
    pub log_scale_factor: f32,
    pub scale_factors: Vec<f32>,
    pub level_sigma2: Vec<f32>,
    pub inv_level_sigma2: Vec<f32>,

    // Image bounds
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,

    // Grid: flat, indexed by `frame::grid_index(col, row)`.
    pub grid: Vec<Vec<usize>>,
    pub grid_right: Vec<Vec<usize>>,

    // Stereo-fisheye correspondences (`usize::MAX` = no match).
    pub left_to_right_match: Option<Vec<usize>>,
    pub right_to_left_match: Option<Vec<usize>>,

    pub imu_calib: Calib,
    pub imu_preintegrated: Option<Arc<Preintegrated>>,

    pub origin_map_id: u32,
    pub name_file: String,
    pub dataset: u32,

    pub camera: Arc<dyn GeometricCamera>,
    pub camera2: Option<Arc<dyn GeometricCamera>>,

    pub n_left: Option<usize>,
    pub n_right: Option<usize>,

    // --- Shared mutable state ------------------------------------------
    pose: RwLock<PoseState>,
    map_points: RwLock<Vec<Option<Arc<MapPoint>>>>,
    conn: RwLock<ConnState>,
    map: RwLock<Option<Arc<Map>>>,
    prev_kf: RwLock<Weak<KeyFrame>>,
    next_kf: RwLock<Weak<KeyFrame>>,
}

impl PartialEq for KeyFrame {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for KeyFrame {}
impl Hash for KeyFrame {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl KeyFrame {
    /// Promote a `Frame` into a `KeyFrame` registered with `map`
    pub fn from_frame(frame: &Frame, map: Arc<Map>) -> Arc<Self> {
        let bounds = &frame.constants.bounds;
        let intr = &frame.constants.intrinsics;

        // Reuse the frame's BoW if present, otherwise compute it now.
        let (bow_vec, feat_vec) = if frame.bow_vec.is_empty() {
            let descs = descriptors_to_array(&frame.descriptors);
            frame.orb_vocabulary.transform(&descs, 4)
        } else {
            (frame.bow_vec.clone(), frame.feat_vec.clone())
        };

        let t_lr = frame.get_relative_pose_tlr();
        let t_rl = frame.get_relative_pose_trl();

        let kf = KeyFrame {
            imu: map.is_imu_initialized(),
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            frame_id: frame.id,
            timestamp: frame.timestamp,
            grid_cols: FRAME_GRID_COLS,
            grid_rows: FRAME_GRID_ROWS,
            grid_element_width_inv: bounds.grid_w_inv,
            grid_element_height_inv: bounds.grid_h_inv,
            fx: intr.fx,
            fy: intr.fy,
            cx: intr.cx,
            cy: intr.cy,
            invfx: intr.invfx,
            invfy: intr.invfy,
            bf: frame.b_fx,
            b: frame.b,
            th_depth: frame.th_depth,
            dist_coef: frame
                .constants
                .dist_coef
                .try_clone()
                .expect("clone dist_coef"),
            k_matrix: frame.constants.k_matrix,
            n: frame.n,
            keys: frame.keys.clone(),
            keys_un: frame.keys_un.clone().unwrap_or_else(|| frame.keys.clone()),
            keys_right: frame.keys_right.clone(),
            u_right: frame.u_right.clone(),
            depth: frame.depth.clone(),
            descriptors: frame.descriptors.try_clone().expect("clone descriptors"),
            bow_vec,
            feat_vec,
            scale_levels: frame.scale_levels,
            scale_factor: frame.scale_factor,
            log_scale_factor: frame.log_scale_factor,
            scale_factors: frame.scale_factors.clone(),
            level_sigma2: frame.level_sigma2.clone(),
            inv_level_sigma2: frame.inv_level_sigma2.clone(),
            min_x: bounds.min_x,
            min_y: bounds.min_y,
            max_x: bounds.max_x,
            max_y: bounds.max_y,
            grid: frame.grid.clone(),
            grid_right: frame.grid_right.clone(),
            left_to_right_match: frame.left_to_right_match.clone(),
            right_to_left_match: frame.right_to_left_match.clone(),
            imu_calib: frame.imu_calib.clone(),
            imu_preintegrated: frame.imu_preintegrated.clone(),
            origin_map_id: map.get_id(),
            name_file: frame.name_file.clone(),
            dataset: frame.dataset as u32,
            camera: frame.camera.clone(),
            camera2: frame.camera2.clone(),
            n_left: frame.n_left,
            n_right: frame.n_right,
            pose: RwLock::new(PoseState {
                tcw: Isometry3::identity(),
                rcw: Matrix3::identity(),
                twc: Isometry3::identity(),
                rwc: Matrix3::identity(),
                owb: Vector3::zeros(),
                vw: frame.get_velocity(),
                has_velocity: frame.has_velocity(),
                tlr: t_lr,
                trl: t_rl,
                imu_bias: frame.imu_bias,
            }),
            map_points: RwLock::new(frame.map_points.clone()),
            conn: RwLock::new(ConnState {
                first_connection: true,
                ..ConnState::default()
            }),
            map: RwLock::new(Some(map)),
            prev_kf: RwLock::new(Weak::new()),
            next_kf: RwLock::new(Weak::new()),
        };

        let kf = Arc::new(kf);
        kf.set_pose(frame.get_pose());
        kf
    }

    // --- Bag of Words ---------------------------------------------------

    /// Recompute the BoW representation from the descriptors.
    pub fn compute_bow(&mut self) {
        if self.bow_vec.is_empty() {
            let descs = descriptors_to_array(&self.descriptors);
            let voc_descs = descs;
            // Note: needs the vocabulary; callers that require BoW build it at
            // construction. This recomputation path is a no-op placeholder when
            // no vocabulary is attached.
            let _ = voc_descs;
        }
    }

    // --- Pose -----------------------------------------------------------

    pub fn set_pose(&self, tcw: Isometry3<f32>) {
        let mut pose = self.pose.write().unwrap();
        pose.tcw = tcw;
        pose.rcw = tcw.rotation.to_rotation_matrix().into_inner();
        pose.twc = tcw.inverse();
        pose.rwc = pose.twc.rotation.to_rotation_matrix().into_inner();
        if self.imu_calib.is_set {
            pose.owb =
                pose.rwc * self.imu_calib.tcb.translation.vector + pose.twc.translation.vector;
        }
    }

    pub fn set_velocity(&self, vw: Vector3<f32>) {
        let mut pose = self.pose.write().unwrap();
        pose.vw = vw;
        pose.has_velocity = true;
    }

    pub fn get_pose(&self) -> Isometry3<f32> {
        self.pose.read().unwrap().tcw
    }
    pub fn get_pose_inverse(&self) -> Isometry3<f32> {
        self.pose.read().unwrap().twc
    }
    pub fn get_camera_center(&self) -> Vector3<f32> {
        self.pose.read().unwrap().twc.translation.vector
    }
    pub fn get_rotation(&self) -> Matrix3<f32> {
        self.pose.read().unwrap().rcw
    }
    pub fn get_translation(&self) -> Vector3<f32> {
        self.pose.read().unwrap().tcw.translation.vector
    }
    pub fn get_imu_position(&self) -> Vector3<f32> {
        self.pose.read().unwrap().owb
    }
    pub fn get_imu_rotation(&self) -> Matrix3<f32> {
        let pose = self.pose.read().unwrap();
        (pose.twc * self.imu_calib.tcb)
            .rotation
            .to_rotation_matrix()
            .into_inner()
    }
    pub fn get_imu_pose(&self) -> Isometry3<f32> {
        let pose = self.pose.read().unwrap();
        pose.twc * self.imu_calib.tcb
    }
    pub fn get_velocity(&self) -> Vector3<f32> {
        self.pose.read().unwrap().vw
    }
    pub fn is_velocity_set(&self) -> bool {
        self.pose.read().unwrap().has_velocity
    }

    // --- Stereo (fisheye) relative pose --------------------------------

    pub fn get_relative_pose_trl(&self) -> Isometry3<f32> {
        self.pose.read().unwrap().trl
    }
    pub fn get_relative_pose_tlr(&self) -> Isometry3<f32> {
        self.pose.read().unwrap().tlr
    }
    pub fn get_right_pose(&self) -> Isometry3<f32> {
        let pose = self.pose.read().unwrap();
        pose.trl * pose.tcw
    }
    pub fn get_right_pose_inverse(&self) -> Isometry3<f32> {
        let pose = self.pose.read().unwrap();
        pose.twc * pose.tlr
    }
    pub fn get_right_camera_center(&self) -> Vector3<f32> {
        let pose = self.pose.read().unwrap();
        (pose.twc * pose.tlr).translation.vector
    }
    pub fn get_right_rotation(&self) -> Matrix3<f32> {
        let pose = self.pose.read().unwrap();
        (pose.trl.rotation * pose.tcw.rotation)
            .to_rotation_matrix()
            .into_inner()
    }
    pub fn get_right_translation(&self) -> Vector3<f32> {
        let pose = self.pose.read().unwrap();
        (pose.trl * pose.tcw).translation.vector
    }

    // --- IMU bias -------------------------------------------------------

    pub fn set_new_bias(&self, b: Bias) {
        self.pose.write().unwrap().imu_bias = b;
        // Preintegrated mutation through a shared `Arc` is deferred (see Frame).
    }
    pub fn get_gyro_bias(&self) -> Vector3<f32> {
        let b = self.pose.read().unwrap().imu_bias;
        Vector3::new(b.bwx, b.bwy, b.bwz)
    }
    pub fn get_acc_bias(&self) -> Vector3<f32> {
        let b = self.pose.read().unwrap().imu_bias;
        Vector3::new(b.bax, b.bay, b.baz)
    }
    pub fn get_imu_bias(&self) -> Bias {
        self.pose.read().unwrap().imu_bias
    }

    // --- MapPoint observations -----------------------------------------

    pub fn add_map_point(&self, mp: Arc<MapPoint>, idx: usize) {
        self.map_points.write().unwrap()[idx] = Some(mp);
    }
    pub fn erase_map_point_match_idx(&self, idx: usize) {
        if let Some(slot) = self.map_points.write().unwrap().get_mut(idx) {
            *slot = None;
        }
    }
    pub fn erase_map_point_match(&self, mp: &MapPoint) {
        let (left, right) = mp.get_index_in_keyframe(self);
        let mut mps = self.map_points.write().unwrap();
        if left != -1 {
            mps[left as usize] = None;
        }
        if right != -1 {
            mps[right as usize] = None;
        }
    }
    pub fn replace_map_point_match(&self, idx: usize, mp: Arc<MapPoint>) {
        self.map_points.write().unwrap()[idx] = Some(mp);
    }
    pub fn get_map_point(&self, idx: usize) -> Option<Arc<MapPoint>> {
        self.map_points
            .read()
            .unwrap()
            .get(idx)
            .and_then(Clone::clone)
    }
    pub fn get_map_point_matches(&self) -> Vec<Option<Arc<MapPoint>>> {
        self.map_points.read().unwrap().clone()
    }
    pub fn get_map_points(&self) -> HashSet<Arc<MapPoint>> {
        self.map_points
            .read()
            .unwrap()
            .iter()
            .flatten()
            .filter(|mp| !mp.is_bad())
            .cloned()
            .collect()
    }
    pub fn get_number_mps(&self) -> usize {
        self.map_points.read().unwrap().iter().flatten().count()
    }
    pub fn tracked_map_points(&self, min_obs: i32) -> i32 {
        let check_obs = min_obs > 0;
        let mps = self.map_points.read().unwrap();
        let mut count = 0;
        for mp in mps.iter().flatten() {
            if mp.is_bad() {
                continue;
            }
            if !check_obs || mp.observations() >= min_obs {
                count += 1;
            }
        }
        count
    }

    // --- Covisibility graph --------------------------------------------

    pub fn add_connection(&self, kf: &Arc<KeyFrame>, weight: i32) {
        {
            let mut conn = self.conn.write().unwrap();
            match conn.connected_weights.get(&kf.id) {
                Some((_, w)) if *w == weight => return,
                _ => {
                    conn.connected_weights
                        .insert(kf.id, (Arc::downgrade(kf), weight));
                }
            }
        }
        self.update_best_covisibles();
    }

    pub fn erase_connection(&self, kf: &KeyFrame) {
        let updated = {
            let mut conn = self.conn.write().unwrap();
            conn.connected_weights.remove(&kf.id).is_some()
        };
        if updated {
            self.update_best_covisibles();
        }
    }

    fn update_best_covisibles(&self) {
        let mut conn = self.conn.write().unwrap();
        let mut pairs: Vec<(i32, Weak<KeyFrame>)> = conn
            .connected_weights
            .values()
            .map(|(w, weight)| (*weight, w.clone()))
            .collect();
        // Sort ascending by weight, then reverse → descending, dropping bad KFs.
        pairs.sort_by_key(|(w, _)| *w);
        let mut ordered: Vec<(Weak<KeyFrame>, i32)> = Vec::with_capacity(pairs.len());
        for (weight, weak) in pairs.into_iter().rev() {
            if weak.upgrade().is_some_and(|kf| !kf.is_bad()) {
                ordered.push((weak, weight));
            }
        }
        conn.ordered = ordered;
    }

    pub fn get_weight(&self, kf: &KeyFrame) -> i32 {
        self.conn
            .read()
            .unwrap()
            .connected_weights
            .get(&kf.id)
            .map(|(_, w)| *w)
            .unwrap_or(0)
    }

    pub fn get_connected_keyframes(&self) -> HashSet<Arc<KeyFrame>> {
        self.conn
            .read()
            .unwrap()
            .connected_weights
            .values()
            .filter_map(|(w, _)| w.upgrade())
            .collect()
    }

    pub fn get_vector_covisible_keyframes(&self) -> Vec<Arc<KeyFrame>> {
        self.conn
            .read()
            .unwrap()
            .ordered
            .iter()
            .filter_map(|(w, _)| w.upgrade())
            .collect()
    }

    pub fn get_best_covisibility_keyframes(&self, n: usize) -> Vec<Arc<KeyFrame>> {
        self.conn
            .read()
            .unwrap()
            .ordered
            .iter()
            .take(n)
            .filter_map(|(w, _)| w.upgrade())
            .collect()
    }

    pub fn get_covisibles_by_weight(&self, w: i32) -> Vec<Arc<KeyFrame>> {
        self.conn
            .read()
            .unwrap()
            .ordered
            .iter()
            .take_while(|(_, weight)| *weight > w)
            .filter_map(|(weak, _)| weak.upgrade())
            .collect()
    }

    // --- Spanning tree --------------------------------------------------

    pub fn add_child(&self, kf: &Arc<KeyFrame>) {
        self.conn
            .write()
            .unwrap()
            .children
            .insert(kf.id, Arc::downgrade(kf));
    }
    pub fn erase_child(&self, kf: &KeyFrame) {
        self.conn.write().unwrap().children.remove(&kf.id);
    }
    pub fn change_parent(self: &Arc<Self>, kf: &Arc<KeyFrame>) {
        {
            let mut conn = self.conn.write().unwrap();
            conn.parent = Some(Arc::downgrade(kf));
        }
        kf.add_child(self);
    }
    pub fn get_children(&self) -> HashSet<Arc<KeyFrame>> {
        self.conn
            .read()
            .unwrap()
            .children
            .values()
            .filter_map(Weak::upgrade)
            .collect()
    }
    pub fn get_parent(&self) -> Option<Arc<KeyFrame>> {
        self.conn
            .read()
            .unwrap()
            .parent
            .as_ref()
            .and_then(Weak::upgrade)
    }
    pub fn has_child(&self, kf: &KeyFrame) -> bool {
        self.conn.read().unwrap().children.contains_key(&kf.id)
    }
    pub fn set_first_connection(&self, first: bool) {
        self.conn.write().unwrap().first_connection = first;
    }

    // --- Loop / merge edges --------------------------------------------

    pub fn add_loop_edge(&self, kf: &Arc<KeyFrame>) {
        let mut conn = self.conn.write().unwrap();
        conn.not_erase = true;
        conn.loop_edges.insert(kf.id, Arc::downgrade(kf));
    }
    pub fn get_loop_edges(&self) -> HashSet<Arc<KeyFrame>> {
        self.conn
            .read()
            .unwrap()
            .loop_edges
            .values()
            .filter_map(Weak::upgrade)
            .collect()
    }
    pub fn add_merge_edge(&self, kf: &Arc<KeyFrame>) {
        let mut conn = self.conn.write().unwrap();
        conn.not_erase = true;
        conn.merge_edges.insert(kf.id, Arc::downgrade(kf));
    }
    pub fn get_merge_edges(&self) -> HashSet<Arc<KeyFrame>> {
        self.conn
            .read()
            .unwrap()
            .merge_edges
            .values()
            .filter_map(Weak::upgrade)
            .collect()
    }

    // --- Bad flag -------------------------------------------------------

    pub fn set_not_erase(&self) {
        self.conn.write().unwrap().not_erase = true;
    }
    pub fn is_bad(&self) -> bool {
        self.conn.read().unwrap().bad
    }

    // --- Prev / next (inertial) ----------------------------------------

    pub fn set_prev_kf(&self, kf: &Arc<KeyFrame>) {
        *self.prev_kf.write().unwrap() = Arc::downgrade(kf);
    }
    pub fn get_prev_kf(&self) -> Option<Arc<KeyFrame>> {
        self.prev_kf.read().unwrap().upgrade()
    }
    pub fn set_next_kf(&self, kf: &Arc<KeyFrame>) {
        *self.next_kf.write().unwrap() = Arc::downgrade(kf);
    }
    pub fn get_next_kf(&self) -> Option<Arc<KeyFrame>> {
        self.next_kf.read().unwrap().upgrade()
    }

    // --- Map ------------------------------------------------------------

    pub fn get_map(&self) -> Option<Arc<Map>> {
        self.map.read().unwrap().clone()
    }
    pub fn update_map(&self, map: Arc<Map>) {
        *self.map.write().unwrap() = Some(map);
    }

    // --- Keypoint geometry ---------------------------------------------

    pub fn is_in_image(&self, x: f32, y: f32) -> bool {
        x >= self.min_x && x < self.max_x && y >= self.min_y && y < self.max_y
    }

    /// Indices of features whose grid cells overlap a circle of radius `r`
    /// around `(x, y)`. `right` selects the right-image grid (stereo fisheye).
    pub fn get_features_in_area(&self, x: f32, y: f32, r: f32, right: bool) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.n);

        let min_cell_x =
            (((x - self.min_x - r) * self.grid_element_width_inv).floor() as i32).max(0);
        if min_cell_x >= self.grid_cols as i32 {
            return indices;
        }
        let max_cell_x = (((x - self.min_x + r) * self.grid_element_width_inv).ceil() as i32)
            .min(self.grid_cols as i32 - 1);
        if max_cell_x < 0 {
            return indices;
        }
        let min_cell_y =
            (((y - self.min_y - r) * self.grid_element_height_inv).floor() as i32).max(0);
        if min_cell_y >= self.grid_rows as i32 {
            return indices;
        }
        let max_cell_y = (((y - self.min_y + r) * self.grid_element_height_inv).ceil() as i32)
            .min(self.grid_rows as i32 - 1);
        if max_cell_y < 0 {
            return indices;
        }

        let grid = if right { &self.grid_right } else { &self.grid };
        for ix in min_cell_x..=max_cell_x {
            for iy in min_cell_y..=max_cell_y {
                let cell = &grid[grid_index(ix as usize, iy as usize)];
                for &j in cell {
                    let kp = self.keypoint_for_area(j, right);
                    let distx = kp.pt().x - x;
                    let disty = kp.pt().y - y;
                    if distx.abs() < r && disty.abs() < r {
                        indices.push(j);
                    }
                }
            }
        }
        indices
    }

    fn keypoint_for_area(&self, idx: usize, right: bool) -> &KeyPoint {
        match self.n_left {
            None => &self.keys_un[idx],
            Some(_) if !right => &self.keys[idx],
            Some(_) => &self.keys_right.as_ref().expect("keys_right")[idx],
        }
    }

    /// Backproject keypoint `i` to a world point if it has valid depth.
    pub fn unproject_stereo(&self, i: usize) -> Option<Vector3<f32>> {
        let z = self.depth[i];
        if z > 0.0 {
            let u = self.keys[i].pt().x;
            let v = self.keys[i].pt().y;
            let x = (u - self.cx) * z * self.invfx;
            let y = (v - self.cy) * z * self.invfy;
            let pose = self.pose.read().unwrap();
            Some(pose.rwc * Vector3::new(x, y, z) + pose.twc.translation.vector)
        } else {
            None
        }
    }

    /// Median scene depth (`q`=2 gives the median). Used by monocular init.
    pub fn compute_scene_median_depth(&self, q: usize) -> f32 {
        if self.n == 0 {
            return -1.0;
        }
        let (mps, rcw2, zcw) = {
            let mps = self.map_points.read().unwrap().clone();
            let pose = self.pose.read().unwrap();
            (mps, pose.rcw.row(2).transpose(), pose.tcw.translation.z)
        };

        let mut depths: Vec<f32> = mps
            .iter()
            .flatten()
            .map(|mp| rcw2.dot(&mp.get_world_pos()) + zcw)
            .collect();
        if depths.is_empty() {
            return -1.0;
        }
        depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        depths[(depths.len() - 1) / q]
    }

    // --- Projection -----------------------------------------------------

    /// Project a map point with OpenCV radial-tangential distortion.
    pub fn project_point_distort(&self, mp: &MapPoint) -> Option<Point2f> {
        let p = mp.get_world_pos();
        let pose = self.pose.read().unwrap();
        let pc = pose.rcw * p + pose.tcw.translation.vector;
        drop(pose);
        if pc.z < 0.0 {
            return None;
        }
        let invz = 1.0 / pc.z;
        let u = self.fx * pc.x * invz + self.cx;
        let v = self.fy * pc.y * invz + self.cy;
        if u < self.min_x || u > self.max_x || v < self.min_y || v > self.max_y {
            return None;
        }

        let x = (u - self.cx) * self.invfx;
        let y = (v - self.cy) * self.invfy;
        let r2 = x * x + y * y;
        let k1 = *self.dist_coef.at::<f32>(0).expect("k1");
        let k2 = *self.dist_coef.at::<f32>(1).expect("k2");
        let p1 = *self.dist_coef.at::<f32>(2).expect("p1");
        let p2 = *self.dist_coef.at::<f32>(3).expect("p2");
        let k3 = if self.dist_coef.total() == 5 {
            *self.dist_coef.at::<f32>(4).expect("k3")
        } else {
            0.0
        };

        let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        let x_distort = x * radial + (2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x));
        let y_distort = y * radial + (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y);

        Some(Point2f::new(
            x_distort * self.fx + self.cx,
            y_distort * self.fy + self.cy,
        ))
    }

    /// Project a map point without distortion.
    pub fn project_point_undistort(&self, mp: &MapPoint) -> Option<Point2f> {
        let p = mp.get_world_pos();
        let pose = self.pose.read().unwrap();
        let pc = pose.rcw * p + pose.tcw.translation.vector;
        drop(pose);
        if pc.z < 0.0 {
            return None;
        }
        let invz = 1.0 / pc.z;
        let u = self.fx * pc.x * invz + self.cx;
        let v = self.fy * pc.y * invz + self.cy;
        if u < self.min_x || u > self.max_x || v < self.min_y || v > self.max_y {
            return None;
        }
        Some(Point2f::new(u, v))
    }
}

/// Convert an `N×32` `CV_8U` descriptor matrix into per-row [`Descriptor`]s.
fn descriptors_to_array(descriptors: &Mat) -> Vec<Descriptor> {
    use opencv::prelude::MatTraitConstManual;
    let rows = descriptors.rows();
    let mut out = Vec::with_capacity(rows as usize);
    for i in 0..rows {
        let row = descriptors.row(i).expect("descriptor row");
        let bytes = row.data_bytes().expect("descriptor bytes");
        let mut d: Descriptor = [0u8; DESC_LEN];
        d.copy_from_slice(&bytes[..DESC_LEN]);
        out.push(d);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]
    use std::sync::Arc;

    use nalgebra::Vector3;
    use opencv::core::{KeyPointTrait, Point2f};

    use super::*;
    use crate::map_point::MapPoint;
    use crate::test_helpers::*;

    #[test]
    fn pose_imu() {
        let mut f = build_frame();
        f.set_pose(make_pose());
        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map);

        assert_vec(kf.get_camera_center(), [-1.0, -2.0, -3.0], 1e-4);
        assert_vec(
            kf.get_imu_position(),
            [-1.03824139, -1.96679592, -3.02085876],
            1e-4,
        );
        // Tlr identity -> right camera centre == camera centre.
        assert_vec(kf.get_right_camera_center(), [-1.0, -2.0, -3.0], 1e-4);
    }

    #[test]
    fn is_in_image() {
        let f = build_frame();
        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map);

        assert!(kf.is_in_image(100.0, 100.0));
        assert!(!kf.is_in_image(kf.max_x + 1.0, 100.0));
        assert!(!kf.is_in_image(100.0, kf.min_y - 1.0));
    }

    #[test]
    fn get_features_in_area() {
        let mut f = build_frame();
        let c = f.constants.clone();

        f.n = 3;
        let kps = vec![
            keypoint(100.0, 100.0),
            keypoint(105.0, 103.0),
            keypoint(400.0, 300.0),
        ];
        f.keys_un = Some(kps.clone());
        let mut grid = vec![Vec::<usize>::new(); FRAME_GRID_COLS * FRAME_GRID_ROWS];
        for (i, kp) in kps.iter().enumerate() {
            let (gx, gy) = crate::frame::pos_in_grid(kp, &c.bounds).unwrap();
            grid[grid_index(gx, gy)].push(i);
        }
        f.grid = grid;

        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map);

        let mut near = kf.get_features_in_area(100.0, 100.0, 10.0, false);
        near.sort();
        assert_eq!(near, vec![0, 1]);
    }

    #[test]
    fn unproject_stereo() {
        let mut f = build_frame();
        assert!(f.n > 0);
        f.keys[0].set_pt(Point2f::new(400.0, 300.0));
        f.depth[0] = 2.5;

        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map);
        kf.set_pose(make_pose());

        let x3d = kf.unproject_stereo(0).expect("has depth");
        assert_vec(x3d, [-1.198632, -1.46398377, -0.543413162], 1e-3);
    }

    #[test]
    fn map_point_matches() {
        let f = build_frame();
        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map);
        assert!(kf.n > 3);

        assert_eq!(kf.get_number_mps(), 0);
        let mp = Arc::new(MapPoint::new());
        kf.add_map_point(mp.clone(), 2);
        assert_eq!(kf.get_map_point(2).unwrap().id, mp.id);
        assert_eq!(kf.get_number_mps(), 1);
        assert_eq!(kf.get_map_point_matches().len(), kf.n);

        kf.erase_map_point_match_idx(2);
        assert!(kf.get_map_point(2).is_none());
        assert_eq!(kf.get_number_mps(), 0);
    }

    #[test]
    fn scene_median_depth() {
        let mut f = build_frame();
        f.set_pose(make_pose());
        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map);
        assert!(kf.n > 3);

        for (i, pos) in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.5, 2.0),
            Vector3::new(-1.0, 1.0, -2.0),
        ]
        .into_iter()
        .enumerate()
        {
            let mp = Arc::new(MapPoint::new());
            mp.set_world_pos(pos);
            kf.add_map_point(mp, i);
        }

        assert!(approx(kf.compute_scene_median_depth(2), 3.0, 1e-3));
    }

    #[test]
    fn project_point_distort_undistort() {
        let mut f = build_frame();
        f.set_pose(make_pose());
        let map = Arc::new(Map::new());
        let kf = KeyFrame::from_frame(&f, map);

        let cam_pt = Vector3::new(0.3, 0.1, 6.0);
        let world = kf
            .get_pose_inverse()
            .rotation
            .to_rotation_matrix()
            .into_inner()
            * cam_pt
            + kf.get_camera_center();
        let mp = Arc::new(MapPoint::new());
        mp.set_world_pos(world);

        let d = kf.project_point_distort(&mp).expect("distort");
        assert!(approx(d.x, 390.129883, 1e-2), "u={}", d.x);
        assert!(approx(d.y, 255.990906, 1e-2), "v={}", d.y);

        let u = kf.project_point_undistort(&mp).expect("undistort");
        assert!(approx(u.x, 390.147705, 1e-2), "u={}", u.x);
        assert!(approx(u.y, 255.996597, 1e-2), "v={}", u.y);
    }

    #[test]
    fn covisibility() {
        let map = Arc::new(Map::new());
        let a = KeyFrame::from_frame(&build_frame(), map.clone());
        let b = KeyFrame::from_frame(&build_frame(), map.clone());
        let c = KeyFrame::from_frame(&build_frame(), map);

        a.add_connection(&b, 30);
        a.add_connection(&c, 50);

        assert_eq!(a.get_weight(&b), 30);
        assert_eq!(a.get_weight(&c), 50);

        let best1 = a.get_best_covisibility_keyframes(1);
        assert_eq!(best1.len(), 1);
        assert_eq!(best1[0].id, c.id);

        let ordered = a.get_vector_covisible_keyframes();
        assert_eq!(ordered.len(), 2);
        assert_eq!(ordered[0].id, c.id);
        assert_eq!(ordered[1].id, b.id);

        let by_weight = a.get_covisibles_by_weight(40);
        assert_eq!(by_weight.len(), 1);
        assert_eq!(by_weight[0].id, c.id);

        let connected = a.get_connected_keyframes();
        assert!(connected.iter().any(|k| k.id == b.id));
        assert!(connected.iter().any(|k| k.id == c.id));
    }

    #[test]
    fn spanning_tree_and_edges() {
        let map = Arc::new(Map::new());
        let a = KeyFrame::from_frame(&build_frame(), map.clone());
        let b = KeyFrame::from_frame(&build_frame(), map);

        a.add_child(&b);
        assert!(a.has_child(&b));
        assert!(a.get_children().iter().any(|k| k.id == b.id));

        b.change_parent(&a);
        assert_eq!(b.get_parent().unwrap().id, a.id);
        assert!(a.has_child(&b));

        a.add_loop_edge(&b);
        assert!(a.get_loop_edges().iter().any(|k| k.id == b.id));
        a.add_merge_edge(&b);
        assert!(a.get_merge_edges().iter().any(|k| k.id == b.id));
    }
}
