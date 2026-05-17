use opencv::core::Mat;
use opencv::prelude::*;
use std::sync::Arc;

use crate::frame::Frame;
use crate::map_point::{self, MapPoint};

pub struct OrbMatcher {
    nn_ratio: f32,
    check_orientation: bool,
}

impl Default for OrbMatcher {
    fn default() -> Self {
        OrbMatcher {
            nn_ratio: 0.6,
            check_orientation: true,
        }
    }
}
impl OrbMatcher {
    pub fn search_by_projection_keypoints_mappoints(
        f: &Frame,
        map_points: Vec<Arc<MapPoint>>,
        th: f32,            // default 3.0
        far_points: bool,   // default false
        th_far_points: f32, // default 50.0f
    ) {
        let mut n_matches = 0;
        let mut left = 0;
        let mut right = 0;
        let factor = th != 1.0;

        for (idx, mp) in map_points.iter().enumerate() {
            if mp.track_in_view && !mp.track_in_view_r {
                continue;
            }
            if far_points && mp.track_depth > th_far_points {
                continue;
            }
            if mp.is_bad() {
                continue;
            }
            if mp.track_in_view {
                let predicted_level = mp.track_scale_level;

                // The size of the window will depend on the viewing direction
                let mut r = radius_by_viewing_cos(mp.track_view_cos);

                if factor {
                    r *= th;
                }

                let indices = f.get_features_in_area(
                    mp.track_proj_x,
                    mp.track_proj_y,
                    r * f.scale_factors[predicted_level as usize],
                    predicted_level - 1,
                    predicted_level,
                    false,
                );

                if !indices.is_empty() {
                    let descriptor = mp.get_descriptor();

                    let mut best_dist = 256;
                    let mut best_level = -1;
                    let mut best_dist2 = 256;
                    let mut best_level2 = -1;
                    let mut best_idx = -1;

                    // Get best and second matches with near keypoints
                    for idx in indices.iter() {
                        if let Some(it) = f.map_points.get(*idx) {
                            // TODO
                        }
                    }
                }

                // TODO
            }
            // TODO
        }
    }
}

// Computes the Hamming distance between two ORB descriptors
fn descriptor_distance(a: &Mat, b: &Mat) -> i32 {
    let pa: &[u8] = a.data_bytes().expect("descriptor a");
    let pb: &[u8] = b.data_bytes().expect("descriptor b");
    pa.iter()
        .zip(pb)
        .map(|(x, y)| (x ^ y).count_ones())
        .sum::<u32>() as i32
}

fn radius_by_viewing_cos(view_cos: f32) -> f32 {
    if view_cos > 0.998 { 2.5 } else { 4.0 }
}
