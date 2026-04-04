use nalgebra::{
    DMatrix, Isometry3, Matrix3, Rotation3, SMatrix, SVD, Translation3, UnitQuaternion, Vector3,
};
use opencv::core::{KeyPoint, Point2f, Point3f};
use opencv::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index;

use crate::geometric_tools::triangulate;

pub struct TwoViewReconstruction {
    // Keypoints reference frame (1)
    keys1: Vec<KeyPoint>,
    // Keypoints current frame (2)
    keys2: Vec<KeyPoint>,
    // Matches from reference to current
    matches12: Vec<(usize, usize)>,
    matched1: Vec<bool>,
    // Calibration
    k: Matrix3<f32>,
    // Standard deviation and variance
    sigma: f32,
    sigma2: f32,
    // Ransac max iterations
    max_iterations: u32,
    // Ransac sets
    sets: Vec<Vec<usize>>,
}

/// Wrapper to send &KeyPoint across thread boundaries.
/// KeyPoint contains a `*mut c_void` from the OpenCV C++ FFI, making it !Sync.
/// The fields we access (pt, size, angle, etc.) are plain data that is safe to read
/// concurrently. We only share immutable references, so no data races should occur in this context.
#[derive(Clone, Copy)]
struct SendKeyPoints<'a>(&'a [KeyPoint]);
unsafe impl Send for SendKeyPoints<'_> {}
unsafe impl Sync for SendKeyPoints<'_> {}

struct ReconstructResult {
    t21: Isometry3<f32>,
    triangulated: Vec<bool>,
}

impl TwoViewReconstruction {
    fn new(k: Matrix3<f32>, sigma: f32, iterations: u32) -> Self {
        TwoViewReconstruction {
            keys1: Vec::new(),
            keys2: Vec::new(),
            matches12: Vec::new(),
            matched1: Vec::new(),
            k: k,
            sigma: sigma,
            sigma2: sigma * sigma,
            max_iterations: iterations,
            sets: Vec::new(),
        }
    }

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    fn reconstruct(
        &mut self,
        keys1: Vec<KeyPoint>,
        keys2: Vec<KeyPoint>,
        matches12: &[Option<usize>],
    ) -> Option<ReconstructResult> {
        self.keys1 = keys1;
        self.keys2 = keys2;

        // Fill structure with current keypoints and matches with reference frame
        self.matches12.clear();
        self.matches12.reserve(self.keys2.len());
        self.matched1.resize(self.keys1.len(), false);
        for (i, matched) in matches12.iter().enumerate() {
            if let Some(&j) = matched.as_ref() {
                self.matches12.push((i, j));
                self.matched1[i] = true;
            } else {
                self.matched1[i] = false;
            }
        }

        let n = self.matches12.len();

        // Generate sets of 8 points for each RANSAC iteration
        self.sets = (0..self.max_iterations).map(|_| vec![0usize; 8]).collect();
        let mut rng = StdRng::seed_from_u64(0);
        for set in &mut self.sets {
            for (j, idx) in index::sample(&mut rng, n, 8).into_iter().enumerate() {
                set[j] = idx;
            }
        }

        let matches12 = &self.matches12;
        let keys1 = SendKeyPoints(&self.keys1);
        let keys2 = SendKeyPoints(&self.keys2);
        let sets = &self.sets;
        let max_iterations = self.max_iterations;
        let sigma = self.sigma;
        let (homography, fundamental) = rayon::join(
            move || find_homography(matches12, keys1, keys2, sets, max_iterations, sigma),
            move || find_fundamental(matches12, keys1, keys2, sets, max_iterations, sigma),
        );

        // Compute ratio of scores
        if homography.score + fundamental.score == 0. {
            return None;
        }
        let rh = homography.score / (homography.score + fundamental.score);

        // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
        const MIN_PARALLAX: f32 = 1.0f32;
        const MIN_TRIANGULATED: usize = 50;
        if rh > 0.5 {
            self.reconstruct_h(homography, MIN_PARALLAX, MIN_TRIANGULATED)
        } else {
            self.reconstruct_f(fundamental, MIN_PARALLAX, MIN_TRIANGULATED)
        }
    }

    fn reconstruct_h(
        &self,
        result: HomographyResult,
        min_parallax: f32,
        min_triangulated: usize,
    ) -> Option<ReconstructResult> {
        let n = result.matches_inliers.iter().filter(|i| **i).count();

        // We recover 8 motion hypotheses using the method of Faugeras et al.
        // Motion and structure from motion in a piecewice planar environment
        // International Journal of Pattern Recognition nad Artificial Intelligence, 1988
        let k_inv = &self.k.try_inverse().unwrap();
        let a = k_inv * result.h21 * &self.k;

        let svd = a.svd(true, true);
        let u = svd.u.unwrap();
        let v_t = svd.v_t.unwrap();
        let v = v_t.transpose();
        let w = svd.singular_values;

        let s = u.determinant() * v_t.determinant();

        let d1 = w[0];
        let d2 = w[1];
        let d3 = w[2];

        if d1 / d2 < 1.00001 || d2 / d3 < 1.00001 {
            return None;
        }

        let mut vr = Vec::<Matrix3<f32>>::with_capacity(8);
        let (mut vt, mut vn) = (
            Vec::<Vector3<f32>>::with_capacity(8),
            Vec::<Vector3<f32>>::with_capacity(8),
        );

        //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
        let aux1 = ((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3)).sqrt();
        let aux3 = ((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3)).sqrt();
        let x1 = [aux1, aux1, -aux1, -aux1];
        let x3 = [aux3, -aux3, aux3, -aux3];

        //case d'=d2
        let aux_stheta = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2).sqrt();

        let ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
        let stheta = [aux_stheta, -aux_stheta, -aux_stheta, aux_stheta];

        for i in 0..4 {
            let mut rp = Matrix3::<f32>::zeros();
            rp[(0, 0)] = ctheta;
            rp[(0, 2)] = -stheta[i];
            rp[(1, 1)] = 1.;
            rp[(2, 0)] = stheta[i];
            rp[(2, 2)] = ctheta;

            let r = s * u * rp * v_t;
            vr.push(r);

            let mut tp = Vector3::<f32>::new(x1[i], 0., -x3[i]);
            tp *= d1 - d3;

            let t = u * tp;
            vt.push(t / t.norm());

            let np = Vector3::<f32>::new(x1[i], 0., x3[i]);

            let mut n = v * np;
            if n[2] < 0. {
                n = -n;
            }
            vn.push(n);
        }

        //case d'=-d2
        let aux_sphi = ((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2).sqrt();
        let cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
        let sphi = [aux_sphi, -aux_sphi, -aux_sphi, aux_sphi];

        for i in 0..4 {
            let mut rp = Matrix3::<f32>::zeros();
            rp[(0, 0)] = cphi;
            rp[(0, 2)] = sphi[i];
            rp[(1, 1)] = -1.;
            rp[(2, 0)] = sphi[i];
            rp[(2, 2)] = -cphi;

            let r = s * u * rp * v_t;
            vr.push(r);

            let mut tp = Vector3::<f32>::new(x1[i], 0., x3[i]);
            tp *= d1 + d3;

            let t = u * tp;
            vt.push(t / t.norm());

            let np = Vector3::<f32>::new(x1[i], 0., x3[i]);

            let mut n = v * np;
            if n[2] < 0. {
                n = -n;
            }
            vn.push(n);
        }

        let mut best_good = 0;
        let mut second_best_good = 0;
        let mut best_solution_idx = 0;
        let mut best_parallax = -1.0f32;
        let mut best_triangulated = Vec::<bool>::new();

        // Instead of applying the visibility constraints proposed in the Faugeras paper (which could fail for points seen with low parallax)
        // We reconstruct all hypotheses and check in terms of triangulated points and parallax
        for i in 0..8 {
            let result = check_rt(
                &vr[i],
                &vt[i],
                &self.keys1,
                &self.keys2,
                &self.matches12,
                &result.matches_inliers,
                &self.k,
                4.0 * self.sigma2,
            );
            if result.num_good > best_good {
                second_best_good = best_good;
                best_good = result.num_good;
                best_solution_idx = i;
                best_parallax = result.parallax;
                best_triangulated = result.good;
            } else if result.num_good > second_best_good {
                second_best_good = result.num_good;
            }
        }

        if second_best_good * 4 < best_good * 3
            && best_parallax >= min_parallax
            && best_good > min_triangulated
            && best_good * 10 > 9 * n
        {
            Some(ReconstructResult {
                t21: pose_from_rt(&vr[best_solution_idx], &vt[best_solution_idx]),
                triangulated: best_triangulated,
            })
        } else {
            None
        }
    }

    fn reconstruct_f(
        &self,
        result: FundamentalResult,
        min_parallax: f32,
        min_triangulated: usize,
    ) -> Option<ReconstructResult> {
        let n = result.matches_inliers.iter().filter(|i| **i).count();

        // Compute essential matrix from fundamental matrix
        let e21 = self.k.transpose() * result.f21 * self.k;

        // Recover the 4 motion hypotheses
        let (r1, r2, t) = decompose_e(&e21);
        let t1 = t;
        let t2 = -t;

        // Reconstruct with the 4 hypotheses and check
        let sigma = self.sigma2 * 4.0;
        let hypotheses = [(&r1, &t1), (&r2, &t1), (&r1, &t2), (&r2, &t2)].map(|(r, t)| {
            (
                check_rt(
                    r,
                    t,
                    &self.keys1,
                    &self.keys2,
                    &self.matches12,
                    &result.matches_inliers,
                    &self.k,
                    sigma,
                ),
                r,
                t,
            )
        });

        let max_good = hypotheses
            .iter()
            .map(|(res, _, _)| res.num_good)
            .max()
            .unwrap();

        let min_good = ((9 * n) / 10).max(min_triangulated);

        let similar = hypotheses
            .iter()
            .filter(|(res, _, _)| res.num_good * 10 > max_good * 7)
            .count();

        // If there is not a clear winner or not enough triangulated points, reject initialization
        if max_good < min_good || similar > 1 {
            return None;
        }

        // Pick the first hypothesis with max_good
        if let Some((chosen, r, t)) = hypotheses
            .iter()
            .find(|(res, _, _)| res.num_good == max_good)
        {
            if chosen.parallax > min_parallax {
                return Some(ReconstructResult {
                    t21: pose_from_rt(r, t),
                    triangulated: chosen.good.clone(),
                });
            }
        }

        None
    }
}

fn normalize(keys: &[KeyPoint]) -> (Vec<Point2f>, Matrix3<f32>) {
    let n = keys.len();
    let mut normalized_points = Vec::<Point2f>::with_capacity(n);

    let (mut mean_x, mut mean_y) = (0.0f32, 0.0f32);
    for it in keys {
        mean_x += it.pt().x;
        mean_y += it.pt().y;
    }
    mean_x = mean_x / n as f32;
    mean_y = mean_y / n as f32;

    let (mut mean_dev_x, mut mean_dev_y) = (0.0f32, 0.0f32);
    for i in 0..n {
        let p = Point2f::new(keys[i].pt().x - mean_x, keys[i].pt().y - mean_y);
        mean_dev_x += p.x.abs();
        mean_dev_y += p.y.abs();
        normalized_points.push(p);
    }
    mean_dev_x = mean_dev_x / n as f32;
    mean_dev_y = mean_dev_y / n as f32;

    let s_x = 1.0 / mean_dev_x;
    let s_y = 1.0 / mean_dev_y;
    for it in normalized_points.iter_mut() {
        it.x = it.x * s_x;
        it.y = it.y * s_y;
    }

    let mut t = Matrix3::<f32>::zeros();
    t[(0, 0)] = s_x;
    t[(1, 1)] = s_y;
    t[(0, 2)] = -mean_x * s_x;
    t[(1, 2)] = -mean_y * s_y;
    t[(2, 2)] = 1.0;

    (normalized_points, t)
}

fn compute_h21(p1: &[Point2f], p2: &[Point2f]) -> Matrix3<f32> {
    let n = p1.len();
    let mut a = DMatrix::<f32>::zeros(2 * n, 9);

    for i in 0..n {
        let u1 = p1[i].x;
        let v1 = p1[i].y;
        let u2 = p2[i].x;
        let v2 = p2[i].y;

        a[(2 * i, 0)] = 0.0;
        a[(2 * i, 1)] = 0.0;
        a[(2 * i, 2)] = 0.0;
        a[(2 * i, 3)] = -u1;
        a[(2 * i, 4)] = -v1;
        a[(2 * i, 5)] = -1.0;
        a[(2 * i, 6)] = v2 * u1;
        a[(2 * i, 7)] = v2 * v1;
        a[(2 * i, 8)] = v2;

        a[(2 * i + 1, 0)] = u1;
        a[(2 * i + 1, 1)] = v1;
        a[(2 * i + 1, 2)] = 1.0;
        a[(2 * i + 1, 3)] = 0.0;
        a[(2 * i + 1, 4)] = 0.0;
        a[(2 * i + 1, 5)] = 0.0;
        a[(2 * i + 1, 6)] = -u2 * u1;
        a[(2 * i + 1, 7)] = -u2 * v1;
        a[(2 * i + 1, 8)] = -u2;
    }

    // A = U Σ Vᵀ; homography is the right singular vector for the smallest singular value.
    // nalgebra stores Vᵀ (`v_t`); column c of V is row c of Vᵀ. With descending σ, that is
    // the last row of `v_t` (Eigen full V, col(8); thin 8×9 case uses col(7) — same index k).
    let svd = a.svd(false, true);
    let v_t = svd.v_t.expect("svd: V^T not computed");
    let k = svd.singular_values.len() - 1;
    Matrix3::from_row_slice(&[
        v_t[(k, 0)],
        v_t[(k, 1)],
        v_t[(k, 2)],
        v_t[(k, 3)],
        v_t[(k, 4)],
        v_t[(k, 5)],
        v_t[(k, 6)],
        v_t[(k, 7)],
        v_t[(k, 8)],
    ])
}

fn compute_f21(p1: &[Point2f], p2: &[Point2f]) -> Matrix3<f32> {
    let n = p1.len();
    let mut a = DMatrix::<f32>::zeros(n, 9);

    for i in 0..n {
        let u1 = p1[i].x;
        let v1 = p1[i].y;
        let u2 = p2[i].x;
        let v2 = p2[i].y;

        a[(i, 0)] = u2 * u1;
        a[(i, 1)] = u2 * v1;
        a[(i, 2)] = u2;
        a[(i, 3)] = v2 * u1;
        a[(i, 4)] = v2 * v1;
        a[(i, 5)] = v2;
        a[(i, 6)] = u1;
        a[(i, 7)] = v1;
        a[(i, 8)] = 1.;
    }

    let svd = a.svd(false, true);
    let v_t = svd.v_t.expect("svd: V^T not computed");
    let k = svd.singular_values.len() - 1;
    let f_pre = Matrix3::from_row_slice(&[
        v_t[(k, 0)],
        v_t[(k, 1)],
        v_t[(k, 2)],
        v_t[(k, 3)],
        v_t[(k, 4)],
        v_t[(k, 5)],
        v_t[(k, 6)],
        v_t[(k, 7)],
        v_t[(k, 8)],
    ]);
    let svd2 = f_pre.svd(true, true);
    let u = svd2.u.expect("svd2: U not computed");
    let v_t2 = svd2.v_t.expect("svd2: V^T not computed");
    let mut w = svd2.singular_values;
    w[2] = 0.0;

    u * Matrix3::from_diagonal(&w) * v_t2
}

struct HomographyResult {
    score: f32,
    matches_inliers: Vec<bool>,
    h21: Matrix3<f32>,
}
fn find_homography(
    matches12: &[(usize, usize)],
    keys1: SendKeyPoints<'_>,
    keys2: SendKeyPoints<'_>,
    sets: &[Vec<usize>],
    max_iterations: u32,
    sigma: f32,
) -> HomographyResult {
    let (keys1, keys2) = (keys1.0, keys2.0);
    let n = matches12.len();

    let (pn1, t1) = normalize(keys1);
    let (pn2, t2) = normalize(keys2);
    let t2inv = t2.try_inverse().unwrap();

    // best results
    let mut score = 0.0f32;
    let mut matches_inliers = vec![false; n];

    // iteration variables
    let (mut pn1_i, mut pn2_i) = (vec![Point2f::default(); 8], vec![Point2f::default(); 8]);
    let (mut h21_i, mut h12_i);
    let mut current_inliers = vec![false; n];
    let mut current_score;
    let mut h21 = Matrix3::<f32>::default();

    // perform all RANSAC iterations and save solution with highest score
    for i in 0..max_iterations as usize {
        // select a minimum set
        for j in 0..8 {
            let idx = sets[i][j];
            pn1_i[j] = pn1[matches12[idx].0];
            pn2_i[j] = pn2[matches12[idx].1];
        }

        let hn = compute_h21(&pn1_i, &pn2_i);
        h21_i = t2inv * hn * t1;
        h12_i = h21_i.try_inverse().unwrap();

        (current_score, current_inliers) =
            check_homography(matches12, keys1, keys2, &h21_i, &h12_i, sigma);

        if current_score > score {
            h21 = h21_i;
            matches_inliers = current_inliers;
            score = current_score;
        }
    }

    HomographyResult {
        score,
        matches_inliers,
        h21,
    }
}

fn check_homography(
    matches12: &[(usize, usize)],
    keys1: &[KeyPoint],
    keys2: &[KeyPoint],
    h21: &Matrix3<f32>,
    h12: &Matrix3<f32>,
    sigma: f32,
) -> (f32, Vec<bool>) {
    const TH: f32 = 5.991;
    let inv_sigma_sq = 1.0 / (sigma * sigma);
    let n = matches12.len();
    let mut score = 0.0f32;
    let mut inliers = vec![false; n];

    for (i, &(i1, i2)) in matches12.iter().enumerate() {
        let u1 = keys1[i1].pt().x;
        let v1 = keys1[i1].pt().y;
        let u2 = keys2[i2].pt().x;
        let v2 = keys2[i2].pt().y;

        // x2in1 = H12 * x2 (reprojection in image 1)
        let w2in1_inv = 1.0 / (h12[(2, 0)] * u2 + h12[(2, 1)] * v2 + h12[(2, 2)]);
        let u2in1 = (h12[(0, 0)] * u2 + h12[(0, 1)] * v2 + h12[(0, 2)]) * w2in1_inv;
        let v2in1 = (h12[(1, 0)] * u2 + h12[(1, 1)] * v2 + h12[(1, 2)]) * w2in1_inv;
        let chi_sq1 = ((u1 - u2in1).powi(2) + (v1 - v2in1).powi(2)) * inv_sigma_sq;

        let mut is_inlier = true;
        if chi_sq1 > TH {
            is_inlier = false;
        } else {
            score += TH - chi_sq1;
        }

        // x1in2 = H21 * x1 (reprojection in image 2)
        let w1in2_inv = 1.0 / (h21[(2, 0)] * u1 + h21[(2, 1)] * v1 + h21[(2, 2)]);
        let u1in2 = (h21[(0, 0)] * u1 + h21[(0, 1)] * v1 + h21[(0, 2)]) * w1in2_inv;
        let v1in2 = (h21[(1, 0)] * u1 + h21[(1, 1)] * v1 + h21[(1, 2)]) * w1in2_inv;
        let chi_sq2 = ((u2 - u1in2).powi(2) + (v2 - v1in2).powi(2)) * inv_sigma_sq;

        if chi_sq2 > TH {
            is_inlier = false;
        } else {
            score += TH - chi_sq2;
        }

        inliers[i] = is_inlier;
    }

    (score, inliers)
}

struct FundamentalResult {
    score: f32,
    matches_inliers: Vec<bool>,
    f21: Matrix3<f32>,
}
fn find_fundamental(
    matches12: &[(usize, usize)],
    keys1: SendKeyPoints<'_>,
    keys2: SendKeyPoints<'_>,
    sets: &[Vec<usize>],
    max_iterations: u32,
    sigma: f32,
) -> FundamentalResult {
    let (keys1, keys2) = (keys1.0, keys2.0);
    let n = matches12.len();

    let (pn1, t1) = normalize(keys1);
    let (pn2, t2) = normalize(keys2);
    let t2t = t2.transpose();

    // best result
    let mut score = 0.0;
    let mut matches_inliers = vec![false; n];

    // iteration variables
    let (mut pn1_i, mut pn2_i) = (vec![Point2f::default(); 8], vec![Point2f::default(); 8]);
    let mut f21i;
    let mut current_inliers = vec![false; n];
    let mut current_score;
    let mut f21 = Matrix3::<f32>::default();

    // perform all RANSAC iterations and save solution with highest score
    for i in 0..max_iterations as usize {
        // select a minimum set
        for j in 0..8 {
            let idx = sets[i][j];
            pn1_i[j] = pn1[matches12[idx].0];
            pn2_i[j] = pn2[matches12[idx].1];
        }

        let f = compute_h21(&pn1_i, &pn2_i);
        f21i = t2t * f * t1;

        (current_score, current_inliers) = check_fundamental(matches12, keys1, keys2, &f21i, sigma);

        if current_score > score {
            f21 = f21i;
            matches_inliers = current_inliers;
            score = current_score;
        }
    }

    FundamentalResult {
        score,
        matches_inliers,
        f21,
    }
}

fn check_fundamental(
    matches12: &[(usize, usize)],
    keys1: &[KeyPoint],
    keys2: &[KeyPoint],
    f21: &Matrix3<f32>,
    sigma: f32,
) -> (f32, Vec<bool>) {
    const TH: f32 = 3.841;
    const TH_SCORE: f32 = 5.991;
    let inv_sigma_sq = 1.0 / (sigma * sigma);
    let n = matches12.len();
    let mut score = 0.0f32;
    let mut inliers = vec![false; n];

    for (i, &(i1, i2)) in matches12.iter().enumerate() {
        let u1 = keys1[i1].pt().x;
        let v1 = keys1[i1].pt().y;
        let u2 = keys2[i2].pt().x;
        let v2 = keys2[i2].pt().y;

        let mut is_inlier = true;

        // l2 = F21 * x1 = (a2, b2, c2) — epipolar line in image 2 / reprojection error
        let a2 = f21[(0, 0)] * u1 + f21[(0, 1)] * v1 + f21[(0, 2)];
        let b2 = f21[(1, 0)] * u1 + f21[(1, 1)] * v1 + f21[(1, 2)];
        let c2 = f21[(2, 0)] * u1 + f21[(2, 1)] * v1 + f21[(2, 2)];
        let num2 = a2 * u2 + b2 * v2 + c2;
        let chi_sq1 = (num2 * num2 / (a2 * a2 + b2 * b2)) * inv_sigma_sq;

        if chi_sq1 > TH {
            is_inlier = false;
        } else {
            score += TH_SCORE - chi_sq1;
        }

        // l1 = x2^T * F21 = (a1, b1, c1) — epipolar line in image 1 / reprojection error
        let a1 = f21[(0, 0)] * u2 + f21[(1, 0)] * v2 + f21[(2, 0)];
        let b1 = f21[(0, 1)] * u2 + f21[(1, 1)] * v2 + f21[(2, 1)];
        let c1 = f21[(0, 2)] * u2 + f21[(1, 2)] * v2 + f21[(2, 2)];
        let num1 = a1 * u1 + b1 * v1 + c1;
        let chi_sq2 = (num1 * num1 / (a1 * a1 + b1 * b1)) * inv_sigma_sq;

        if chi_sq2 > TH {
            is_inlier = false;
        } else {
            score += TH_SCORE - chi_sq2;
        }

        inliers[i] = is_inlier;
    }

    (score, inliers)
}

struct RTResult {
    good: Vec<bool>,
    num_good: usize,
    p3d: Vec<Point3f>,
    parallax: f32,
}
fn check_rt(
    r: &Matrix3<f32>,
    t: &Vector3<f32>,
    keys1: &[KeyPoint],
    keys2: &[KeyPoint],
    matches12: &[(usize, usize)],
    matches_inliers: &[bool],
    k: &Matrix3<f32>,
    th2: f32,
) -> RTResult {
    // Calibration parameters
    let fx = k[(0, 0)];
    let fy = k[(1, 1)];
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];

    let mut good = vec![false; keys1.len()];
    let mut p3d = Vec::with_capacity(keys1.len());

    let mut v_cos_parallax = Vec::<f32>::with_capacity(keys1.len());

    // camera 1 projection matrix k[I|0]
    let mut p1 = SMatrix::<f32, 3, 4>::zeros();
    p1.fixed_view_mut::<3, 3>(0, 0).copy_from(&k);

    let o1 = Vector3::<f32>::zeros();

    // camera 2 projection matrix k[R|t]
    let mut p2 = SMatrix::<f32, 3, 4>::zeros();
    p2.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    p2.fixed_view_mut::<3, 1>(0, 3).copy_from(&t);
    p2 = k * p2;

    let o2 = -r.transpose() * t;

    let mut num_good = 0;

    for i in 0..matches12.len() {
        if !matches_inliers[i] {
            continue;
        }
        let kp1 = &keys1[matches12[i].0];
        let kp2 = &keys2[matches12[i].1];
        let x_p1 = Vector3::<f32>::new(kp1.pt().x, kp1.pt().y, 1.);
        let x_p2 = Vector3::<f32>::new(kp2.pt().x, kp2.pt().y, 1.);

        let Some(p3dc1) = triangulate(&x_p1, &x_p2, &p1, &p2) else {
            good[matches12[i].0] = false;
            continue;
        };
        if !p3dc1[0].is_finite() || !p3dc1[1].is_finite() || !p3dc1[2].is_finite() {
            good[matches12[i].0] = false;
            continue;
        }

        // check parallax
        let normal1 = p3dc1 - o1;
        let dist1 = normal1.norm();
        let normal2 = p3dc1 - o2;
        let dist2 = normal2.norm();

        let cos_parallax = normal1.dot(&normal2) / (dist1 * dist2);

        // check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if p3dc1[2] <= 0. && cos_parallax < 0.99998 {
            continue;
        }
        // check depth in front of second camera
        let p3dc2 = r * p3dc1 + t;
        if p3dc2[2] <= 0. && cos_parallax < 0.99998 {
            continue;
        }

        // check reprojection error in first image
        let inv_z1 = 1.0 / p3dc1[2];
        let im1x = fx * p3dc1[0] * inv_z1 + cx;
        let im1y = fy * p3dc1[1] * inv_z1 + cy;

        let square_error =
            (im1x - kp1.pt().x) * (im1x - kp1.pt().x) + (im1y - kp1.pt().y) * (im1y - kp1.pt().y);
        if square_error > th2 {
            continue;
        }

        // check reprojection error in second image
        let inv_z2 = 1.0 / p3dc2[2];
        let im2x = fx * p3dc2[0] * inv_z2 + cx;
        let im2y = fy * p3dc2[1] * inv_z2 + cy;
        let square_error =
            (im2x - kp2.pt().x) * (im2x - kp2.pt().x) + (im2y - kp2.pt().y) * (im2y - kp2.pt().y);
        if square_error > th2 {
            continue;
        }

        v_cos_parallax.push(cos_parallax);
        p3d[matches12[i].0] = Point3f::new(p3dc1[0], p3dc1[1], p3dc1[2]);
        num_good += 1;

        if cos_parallax < 0.99998 {
            good[matches12[i].0] = true;
        }
    }

    let parallax = if num_good > 0 {
        v_cos_parallax.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = 50usize.min(v_cos_parallax.len() - 1);
        v_cos_parallax[idx].acos() * 180.0 / std::f32::consts::PI
    } else {
        0.0f32
    };

    RTResult {
        good,
        num_good,
        p3d,
        parallax,
    }
}

fn decompose_e(e: &Matrix3<f32>) -> (Matrix3<f32>, Matrix3<f32>, Vector3<f32>) {
    let svd = SVD::new(*e, true, true);

    let u = svd.u.expect("SVD requested U but none was returned");
    let v_t = svd.v_t.expect("SVD requested V^T but none was returned");

    let mut t = u.column(2).into_owned().normalize();

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let mut r1 = u * w * v_t;
    if r1.determinant() < 0.0 {
        r1 = -r1;
    }

    let mut r2 = u * w.transpose() * v_t;
    if r2.determinant() < 0.0 {
        r2 = -r2;
    }

    (r1, r2, t)
}

fn pose_from_rt(r: &nalgebra::Matrix3<f32>, t: &nalgebra::Vector3<f32>) -> Isometry3<f32> {
    Isometry3::from_parts(
        Translation3::from(*t),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r)),
    )
}
