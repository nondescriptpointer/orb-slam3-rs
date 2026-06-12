use opencv::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;

use crate::frame::Frame;
use crate::key_frame::KeyFrame;
use crate::map_point::MapPoint;
use nalgebra::Point3;

const TH_HIGH: i32 = 100;
const TH_LOW: i32 = 50;
const HISTO_LENGTH: usize = 30;

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
    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    pub fn search_by_projection_keypoints_mappoints(
        &self,
        f: &mut Frame,
        map_points: Vec<Arc<MapPoint>>,
        th: f32,            // default 3.0
        far_points: bool,   // default false
        th_far_points: f32, // default 50.0f
    ) -> i32 {
        let mut n_matches = 0;
        let factor = th != 1.0;

        for mp in map_points.iter() {
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

                        // Also match with the stereo observation at right camera
                        if let (Some(_n_left), Some(right_to_left_match), Some(best_idx)) =
                            (&f.n_left, &f.right_to_left_match, best_idx)
                        {
                            if let Some(m) = right_to_left_match.get(best_idx) {
                                if let Some(p) = f.map_points.get_mut(*m) {
                                    *p = Some(mp.clone());
                                    n_matches += 1;
                                }
                            }
                        }

                        if let (Some(best_idx), Some(n_left)) = (best_idx, f.n_left) {
                            if let Some(p) = f.map_points.get_mut(best_idx + n_left) {
                                *p = Some(mp.clone());
                                n_matches += 1;
                            }
                        }
                    }
                }
            }
        }
        n_matches
    }

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    pub fn search_by_projection_last_frame(
        &self,
        current_frame: &mut Frame,
        last_frame: &Frame,
        th: f32, // default 3.0
        mono: bool,
    ) -> i32 {
        let mut n_matches = 0;

        // Rotation Histogram (to check rotation consistency)
        let mut rot_hist: [Vec<usize>; HISTO_LENGTH] =
            std::array::from_fn(|_| Vec::with_capacity(500));
        let factor = 1.0 / HISTO_LENGTH as f32;

        let t_cw = current_frame.get_pose();
        let t_wc = t_cw.inverse();
        let twc = t_wc.translation.vector;

        let t_lw = last_frame.get_pose();
        let tlc = t_lw * twc;

        let forward = tlc.z > current_frame.b && !mono;
        let backward = -tlc.z > current_frame.b && !mono;

        for i in 0..last_frame.n {
            let mp = last_frame.map_points.get(i);
            if let Some(Some(mp)) = mp {
                if !last_frame.outlier[i] {
                    // Project
                    let x_3dw = mp.get_world_pos();
                    let x_3dc = t_cw * x_3dw;

                    let xc = x_3dc[0];
                    let yc = x_3dc[1];
                    let inv_zc = 1.0 / x_3dc[2];

                    if inv_zc < 0. {
                        continue;
                    }

                    let uv = current_frame.camera.project_n(&Point3::from(x_3dc));

                    let bounds = current_frame.constants.bounds;
                    if uv[0] < bounds.min_x || uv[0] > bounds.max_x {
                        continue;
                    }
                    if uv[1] < bounds.min_y || uv[1] > bounds.max_y {
                        continue;
                    }

                    let last_octave =
                        if last_frame.n_left.is_none() || i < last_frame.n_left.unwrap() {
                            last_frame.keys[i].octave()
                        } else {
                            last_frame.keys_right.as_ref().expect("missing keys")
                                [i - last_frame.n_left.expect("missing n left")]
                            .octave()
                        };

                    // Search in a window, size depends on scale
                    let radius = th * current_frame.scale_factors[last_octave as usize];

                    let indices2 = {
                        let (min_level, max_level) = if forward {
                            (last_octave, -1)
                        } else if backward {
                            (0, last_octave)
                        } else {
                            (last_octave - 1, last_octave + 1)
                        };
                        current_frame
                            .get_features_in_area(uv[0], uv[1], radius, min_level, max_level, false)
                    };

                    if indices2.is_empty() {
                        continue;
                    }

                    let d_mp = mp.get_descriptor();

                    let mut best_dist = 256;
                    let mut best_idx2 = 0;

                    for i2 in indices2 {
                        if let Some(Some(mp)) = current_frame.map_points.get(i2) {
                            if mp.observations() > 0 {
                                continue;
                            }
                        }
                        if current_frame.n_left.is_none() && current_frame.u_right[i2] > 0. {
                            let ur = uv[0] - current_frame.b_fx * inv_zc;
                            let er = (ur - current_frame.u_right[i2]).abs();
                            if er > radius {
                                continue;
                            }
                        }
                        let d = current_frame
                            .descriptors
                            .row(i2 as i32)
                            .expect("missing descriptor");
                        let dist = descriptor_distance(&d_mp, &d);
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx2 = i2;
                        }
                    }

                    if best_dist < TH_HIGH {
                        *current_frame.map_points.get_mut(best_idx2).unwrap() = Some(mp.clone());
                        n_matches += 1;

                        if self.check_orientation {
                            let kp_lf = if let Some(n_left) = last_frame.n_left {
                                if i < n_left {
                                    last_frame.keys.get(i).unwrap()
                                } else {
                                    last_frame
                                        .keys_right
                                        .as_ref()
                                        .expect("missing keys right")
                                        .get(i - n_left)
                                        .unwrap()
                                }
                            } else {
                                last_frame
                                    .keys_un
                                    .as_ref()
                                    .expect("missing keys un")
                                    .get(i)
                                    .unwrap()
                            };
                            let kp_cf = if let Some(n_left) = current_frame.n_left {
                                if i < n_left {
                                    current_frame.keys.get(i).unwrap()
                                } else {
                                    current_frame
                                        .keys_right
                                        .as_ref()
                                        .expect("missing keys right")
                                        .get(i - n_left)
                                        .unwrap()
                                }
                            } else {
                                current_frame
                                    .keys_un
                                    .as_ref()
                                    .expect("missing keys un")
                                    .get(best_idx2)
                                    .unwrap()
                            };

                            let mut rot = kp_lf.angle() - kp_cf.angle();
                            if rot < 0. {
                                rot += 360.;
                            }
                            let mut bin = (rot * factor).round() as usize;
                            if bin == HISTO_LENGTH {
                                bin = 0;
                            }
                            assert!(bin >= 0 && bin < HISTO_LENGTH);
                            rot_hist[bin].push(best_idx2);
                        }
                    }
                    if let Some(n_left) = current_frame.n_left {
                        let x_3dr = current_frame.get_relative_pose_trl() * x_3dc;
                        let x_3dc_point = Point3::from(x_3dr);
                        let uv = current_frame.camera.project_n(&x_3dc_point);

                        let last_octave =
                            if last_frame.n_left.is_none() || i < last_frame.n_left.unwrap() {
                                last_frame.keys[i].octave()
                            } else {
                                last_frame.keys_right.as_ref().expect("missing keys")
                                    [i - last_frame.n_left.expect("missing n left")]
                                .octave()
                            };

                        // Search in a window. Size depend on scale
                        let radius = th * current_frame.scale_factors[last_octave as usize];

                        let indices2 = {
                            let (min_level, max_level) = if forward {
                                (last_octave, -1)
                            } else if backward {
                                (0, last_octave)
                            } else {
                                (last_octave - 1, last_octave + 1)
                            };
                            current_frame.get_features_in_area(
                                uv[0], uv[1], radius, min_level, max_level, true,
                            )
                        };

                        let d_mp = mp.get_descriptor();

                        let mut best_dist = 256;
                        let mut best_idx2 = 0;

                        for i2 in indices2 {
                            if let Some(Some(mp)) = current_frame.map_points.get(i2 + n_left) {
                                if mp.observations() > 0 {
                                    continue;
                                }
                            }
                            let d = current_frame
                                .descriptors
                                .row((i2 + n_left) as i32)
                                .expect("missing descriptor");
                            let dist = descriptor_distance(&d_mp, &d);
                            if dist < best_dist {
                                best_dist = dist;
                                best_idx2 = i2;
                            }
                        }
                        if best_dist < TH_HIGH {
                            *current_frame
                                .map_points
                                .get_mut(best_idx2 + n_left)
                                .unwrap() = Some(mp.clone());
                            n_matches += 1;

                            if self.check_orientation {
                                let kp_lf = if let Some(n_left) = last_frame.n_left {
                                    if i < n_left {
                                        last_frame.keys.get(i).unwrap()
                                    } else {
                                        last_frame
                                            .keys_right
                                            .as_ref()
                                            .expect("missing keys right")
                                            .get(i - n_left)
                                            .unwrap()
                                    }
                                } else {
                                    last_frame
                                        .keys_un
                                        .as_ref()
                                        .expect("missing keys un")
                                        .get(i)
                                        .unwrap()
                                };
                                let kp_cf = current_frame
                                    .keys_right
                                    .as_ref()
                                    .unwrap()
                                    .get(best_idx2)
                                    .unwrap();

                                let mut rot = kp_lf.angle() - kp_cf.angle();
                                if rot < 0. {
                                    rot += 360.;
                                }
                                let mut bin = (rot * factor).round() as usize;
                                if bin == HISTO_LENGTH {
                                    bin = 0;
                                }
                                assert!(bin >= 0 && bin < HISTO_LENGTH);
                                rot_hist[bin].push(best_idx2 + n_left);
                            }
                        }
                    }
                }
            }
        }

        // Apply rotation consistency
        if self.check_orientation {
            let maxima = compute_three_maxima(&rot_hist);
            for i in 0..HISTO_LENGTH {
                if Some(i) != maxima[0] && Some(i) != maxima[1] && Some(i) != maxima[2] {
                    for j in 0..rot_hist[i].len() {
                        current_frame.map_points[rot_hist[i][j]] = None;
                        n_matches -= 1;
                    }
                }
            }
        }

        n_matches
    }

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    pub fn search_by_projection_keyframe(
        &self,
        current_frame: &mut Frame,
        kf: &KeyFrame,
        already_found: HashSet<Arc<MapPoint>>,
        th: f32, // default 3.0
        orb_dist: i32,
    ) -> i32 {
        let mut n_matches = 0;

        let t_cw = current_frame.get_pose();
        let t_wc = t_cw.inverse();
        let ow = t_wc.translation.vector;

        // Rotation Histogram (to check rotation consistency)
        let mut rot_hist: [Vec<usize>; HISTO_LENGTH] =
            std::array::from_fn(|_| Vec::with_capacity(500));
        let factor = 1.0 / HISTO_LENGTH as f32;

        let mps = kf.get_map_point_matches();

        for (i, mp) in mps.into_iter().enumerate() {
            if let Some(mp) = mp
                && !mp.is_bad()
                && !already_found.contains(&mp)
            {
                // Project
                let x_3dw = mp.get_world_pos();
                let x_3dc = t_cw * x_3dw;

                let uv = current_frame.camera.project_n(&Point3::from(x_3dc));

                let bounds = current_frame.constants.bounds;
                if uv[0] < bounds.min_x || uv[0] > bounds.max_x {
                    continue;
                }
                if uv[1] < bounds.min_y || uv[1] > bounds.max_y {
                    continue;
                }

                // Compute predicted scale level
                let po = x_3dw - ow;
                let dist3d = po.norm();

                let max_distance = mp.get_max_distance_invariance();
                let min_distance = mp.get_min_distance_invariance();

                // Depth must be inside the scale pyramid of the image
                if dist3d < min_distance || dist3d > max_distance {
                    continue;
                }

                let predicted_level = mp.predict_scale(dist3d, current_frame);

                // Search in a window
                let radius = th * current_frame.scale_factors[predicted_level];

                let indices2 = current_frame.get_features_in_area(
                    uv[0],
                    uv[1],
                    radius,
                    predicted_level as i32 - 1,
                    predicted_level as i32 + 1,
                    false,
                );
                if indices2.is_empty() {
                    continue;
                }

                let d_mp = mp.get_descriptor();

                let mut best_dist = 256;
                let mut best_idx2 = 0;

                for i2 in indices2 {
                    if current_frame
                        .map_points
                        .get(i2)
                        .as_ref()
                        .is_none_or(|inner| inner.is_none())
                    {
                        continue;
                    }

                    let d = current_frame.descriptors.row(i2 as i32).unwrap();

                    let dist = descriptor_distance(&d_mp, &d);

                    if dist < best_dist {
                        best_dist = dist;
                        best_idx2 = i2;
                    }
                }

                if best_dist <= orb_dist {
                    if let Some(p) = current_frame.map_points.get_mut(best_idx2) {
                        *p = Some(mp);
                        n_matches += 1;
                    }

                    if self.check_orientation {
                        let mut rot = kf.keys_un.get(i).unwrap().angle()
                            - current_frame
                                .keys_un
                                .as_ref()
                                .unwrap()
                                .get(best_idx2)
                                .unwrap()
                                .angle();
                        if rot < 0. {
                            rot += 360.;
                        }
                        let mut bin = (rot * factor).round() as usize;
                        if bin == HISTO_LENGTH {
                            bin = 0;
                        }
                        assert!(bin >= 0 && bin < HISTO_LENGTH);
                        rot_hist[bin].push(best_idx2);
                    }
                }
            }
        }

        // Apply rotation consistency
        if self.check_orientation {
            let maxima = compute_three_maxima(&rot_hist);
            for i in 0..HISTO_LENGTH {
                if Some(i) != maxima[0] && Some(i) != maxima[1] && Some(i) != maxima[2] {
                    for j in 0..rot_hist[i].len() {
                        current_frame.map_points[rot_hist[i][j]] = None;
                        n_matches -= 1;
                    }
                }
            }
        }

        n_matches
    }
}

// Computes the Hamming distance between two ORB descriptors
// Original code uses bit-twiddling here but count_ones() should be efficient as is
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

fn compute_three_maxima<T>(histo: &[Vec<T>]) -> [Option<usize>; 3] {
    let mut top = [(0usize, None); 3];
    for (i, bucket) in histo.iter().enumerate() {
        let item = (bucket.len(), Some(i));
        if item.0 > top[0].0 {
            top = [item, top[0], top[1]];
        } else if item.0 > top[1].0 {
            top = [top[0], item, top[1]];
        } else if item.0 > top[2].0 {
            top[2] = item;
        }
    }
    let threshold = top[0].0 / 10;
    [
        top[0].1,
        top[1].1.filter(|_| top[1].0 >= threshold),
        top[2].1.filter(|_| top[2].0 >= threshold),
    ]
}
