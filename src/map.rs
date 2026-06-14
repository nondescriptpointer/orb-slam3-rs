use std::collections::HashSet;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};

static NEXT_MAP_ID: AtomicU32 = AtomicU32::new(0);

/// Shared map state. Cheap-to-clone identity fields are stored inline; the
/// mutable membership/flags live behind a single `RwLock`.
pub struct Map {
    id: u32,
    init_kf_id: u64,
    inner: RwLock<MapInner>,
}

#[derive(Default)]
struct MapInner {
    imu_initialized: bool,
    map_points: HashSet<usize>,
    key_frames: HashSet<u64>,
}

impl Default for Map {
    fn default() -> Self {
        Self::new()
    }
}

impl Map {
    /// Create a new map with a fresh, process-unique id.
    pub fn new() -> Self {
        Map {
            id: NEXT_MAP_ID.fetch_add(1, Ordering::SeqCst),
            init_kf_id: 0,
            inner: RwLock::new(MapInner::default()),
        }
    }

    /// Create a new map whose initial KeyFrame id is `init_kf_id`.
    pub fn with_init_kf_id(init_kf_id: u64) -> Self {
        Map {
            id: NEXT_MAP_ID.fetch_add(1, Ordering::SeqCst),
            init_kf_id,
            inner: RwLock::new(MapInner::default()),
        }
    }

    pub fn get_id(&self) -> u32 {
        self.id
    }

    pub fn get_init_kf_id(&self) -> u64 {
        self.init_kf_id
    }

    pub fn is_imu_initialized(&self) -> bool {
        self.inner.read().unwrap().imu_initialized
    }

    pub fn set_imu_initialized(&self, value: bool) {
        self.inner.write().unwrap().imu_initialized = value;
    }

    pub fn add_map_point(&self, id: usize) {
        self.inner.write().unwrap().map_points.insert(id);
    }

    pub fn erase_map_point(&self, id: usize) {
        self.inner.write().unwrap().map_points.remove(&id);
    }

    pub fn add_key_frame(&self, id: u64) {
        self.inner.write().unwrap().key_frames.insert(id);
    }

    pub fn erase_key_frame(&self, id: u64) {
        self.inner.write().unwrap().key_frames.remove(&id);
    }

    /// Number of (non-erased) map points currently registered with the map.
    pub fn map_points_in_map(&self) -> usize {
        self.inner.read().unwrap().map_points.len()
    }

    /// Number of (non-erased) keyframes currently registered with the map.
    pub fn key_frames_in_map(&self) -> usize {
        self.inner.read().unwrap().key_frames.len()
    }
}
