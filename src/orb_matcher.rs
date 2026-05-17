use opencv::core::Mat;
use opencv::prelude::*;
use std::sync::Arc;

use crate::frame::Frame;
use crate::map_point::{self, MapPoint};

const TH_HIGH: i32 = 100;
const TH_LOW: i32 = 50;
const HISTO_LENGTH: i32 = 30;

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
        &self,
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
                    let mp_descriptor = mp.get_descriptor();

                    let mut best_dist = 256;
                    let mut best_level = None;
                    let mut best_dist2 = 256;
                    let mut best_level2 = None;
                    let mut best_idx = None;

                    // Get best and second matches with near keypoints
                    for idx in indices.into_iter() {
                        if let Some(Some(it)) = f.map_points.get(idx) {
                            if it.observations() > 0 {
                                continue;
                            }
                        }

                        if f.n_left.is_none() && f.u_right.get(idx).is_some_and(|&v| v > 0.0) {
                            let er = (mp.track_proj_xr - f.u_right[idx]).abs();
                            if er > r * f.scale_factors[predicted_level as usize] {
                                continue;
                            }
                        }

                        let d = f.descriptors.row(idx as i32).expect("get row");

                        let dist = descriptor_distance(&mp_descriptor, &d);

                        let octave_at = |idx: usize| -> i32 {
                            if f.n_left.is_none() {
                                f.keys_un.as_ref().unwrap()[idx].octave()
                            } else {
                                if idx < f.n_left.unwrap() {
                                    f.keys[idx].octave()
                                } else {
                                    f.keys_right.as_ref().unwrap()[idx - f.n_left.unwrap()].octave()
                                }
                            }
                        };
                        if dist < best_dist {
                            best_dist2 = best_dist;
                            best_dist = dist;
                            best_level2 = best_level;
                            best_level = Some(octave_at(idx));
                            best_idx = Some(idx);
                        } else if dist < best_dist2 {
                            best_level2 = Some(octave_at(idx));
                            best_dist2 = dist;
                        }
                    }

                    // Apply ratio to second match (only if best and second are in the same scale level)
                    if best_dist <= TH_HIGH {
                        if best_level == best_level2
                            && best_dist as f32 > self.nn_ratio * best_dist2 as f32
                        {
                            continue;
                        }
                        // TODO
                    }
                }

                // TODO
            }
            // TODO
        }
    }
}

// Computes the Hamming distance between two ORB descriptors
fn descriptor_distance(a: &impl MatTraitConst, b: &impl MatTraitConst) -> i32 {
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
