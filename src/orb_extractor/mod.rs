mod brief_pattern;

use opencv::core::{KeyPoint, Mat, Point, Point2i};

use crate::compat::cv_round_f32;

const PATCH_SIZE: u32 = 31;
const HALF_PATCH_SIZE: u32 = 15;
const EDGE_THRESHOLD: u32 = 19;

#[derive(Default)]
struct ExtractorNode {
    pub keys: Vec<KeyPoint>,
    pub ul: Point2i,
    pub ur: Point2i,
    pub bl: Point2i,
    pub br: Point2i,
    pub no_more: bool,
}

#[derive(Default)]
pub struct OrbExtractor {
    pub image_pyramid: Vec<Mat>,
    pattern: Vec<Point>,
    features: usize,
    scale_factor: f64,
    levels: usize,
    ini_th_fast: usize,
    min_th_fast: usize,
    features_per_level: Vec<usize>,
    umax: Vec<u32>,
    v_scale_factor: Vec<f32>,
    v_inv_scale_factor: Vec<f32>,
    v_level_sigma2: Vec<f32>,
    v_inv_level_sigma2: Vec<f32>,
}

pub enum Score {
    Harris,
    Fast,
}

impl OrbExtractor {
    pub fn new(
        features: usize,
        scale_factor: f32,
        levels: usize,
        ini_th_fast: usize,
        min_th_fast: usize,
    ) -> Self {
        let mut v_scale_factor = vec![0.; levels];
        let mut v_level_sigma2 = vec![0.; levels];
        v_scale_factor[0] = 1.;
        v_level_sigma2[0] = 1.;
        for i in 1..levels {
            v_scale_factor[i] = v_scale_factor[i - 1] * scale_factor;
            v_level_sigma2[i] = v_scale_factor[i] * v_scale_factor[i];
        }

        let mut v_inv_scale_factor = vec![0.; levels];
        let mut v_inv_level_sigma2 = vec![0.; levels];
        for i in 0..levels {
            v_inv_scale_factor[i] = 1. / v_scale_factor[i];
            v_inv_level_sigma2[i] = 1. / v_level_sigma2[i];
        }

        let image_pyramid = Vec::with_capacity(levels);

        let mut features_per_level = Vec::with_capacity(levels);
        let inv_factor = 1. / scale_factor;
        let mut desired_features_per_scale =
            features as f32 * (1. - inv_factor) / (1.0 - inv_factor.powf(levels as f32));

        let mut sum_features = 0;
        for _ in 0..(levels - 1) {
            let val = cv_round_f32(desired_features_per_scale);
            sum_features += val;
            features_per_level.push(val as usize);
            desired_features_per_scale *= inv_factor;
        }
        features_per_level.push((features as i32 - sum_features).max(0) as usize);

        let points: u32 = 512;

        OrbExtractor {
            features,
            scale_factor: scale_factor as f64,
            levels,
            ini_th_fast,
            min_th_fast,
            v_scale_factor,
            v_level_sigma2,
            v_inv_scale_factor,
            v_inv_level_sigma2,
            image_pyramid,
            features_per_level,
            ..Default::default()
        }
    }
}
