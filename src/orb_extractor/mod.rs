mod brief_pattern;

use opencv::core::{
    _InputArrayTraitConst, CV_8UC1, InputArray, KeyPoint, Mat, MatTraitConst, OutputArray, Point2i,
    Scalar, Size,
};

use crate::{
    compat::{cv_ceil_f32, cv_floor_f32, cv_round_f32},
    orb_extractor::brief_pattern::{BRIEF_PATTERN, BriefPair},
};

const PATCH_SIZE: usize = 31;
const HALF_PATCH_SIZE: usize = 15;
const EDGE_THRESHOLD: i32 = 19;

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
    pub image_pyramid: Option<Vec<Mat>>,
    pattern: Vec<BriefPair>,
    features: usize,
    scale_factor: f64,
    levels: usize,
    ini_th_fast: usize,
    min_th_fast: usize,
    features_per_level: Vec<usize>,
    umax: Vec<i32>,
    v_scale_factor: Vec<f32>,
    v_inv_scale_factor: Vec<f32>,
    v_level_sigma2: Vec<f32>,
    v_inv_level_sigma2: Vec<f32>,
}

pub enum Score {
    Harris,
    Fast,
}

pub enum ExtractionError {
    EmptyImage,
    InvalidInput,
    InvalidInputType,
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

        let pattern = BRIEF_PATTERN.into();

        // This is for orientation
        // Pre-compute the end of a row in a sircular patch
        let mut umax = vec![0; HALF_PATCH_SIZE + 1];
        let vmax = cv_floor_f32(HALF_PATCH_SIZE as f32 * 2.0_f32.sqrt() / 2. + 1.);
        let vmin = cv_ceil_f32(HALF_PATCH_SIZE as f32 * 2.0_f32.sqrt() / 2.);
        let hp2 = (HALF_PATCH_SIZE * HALF_PATCH_SIZE) as i32;
        for v in 0..=vmax {
            umax[v as usize] = cv_round_f32(((hp2 - v * v) as f32).sqrt())
        }

        // Make sure we are symmetric
        let mut v0: i32 = 0;
        for v in (vmin..=HALF_PATCH_SIZE as i32).rev() {
            while umax[v0 as usize] == umax[(v0 + 1) as usize] {
                v0 += 1;
            }
            umax[v as usize] = v0;
            v0 += 1;
        }

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
            features_per_level,
            pattern,
            umax,
            ..Default::default()
        }
    }

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation
    pub fn compute(
        &self,
        image: InputArray,
        mask: InputArray,
        keypoints: &Vec<KeyPoint>,
        descriptors: OutputArray,
        lapping_area: Vec<i32>,
    ) -> Result<(), ExtractionError> {
        if image.empty().unwrap_or(true) {
            return Err(ExtractionError::EmptyImage);
        }

        let image_mat = image
            .get_mat_def()
            .map_err(|_| ExtractionError::InvalidInput)?;
        if image_mat.typ() != CV_8UC1 {
            return Err(ExtractionError::InvalidInputType);
        }

        // Pre-compute the scale pyramid
        // TODO: here

        Ok(())
    }

    fn compute_pyramid(&mut self, image: Mat) {
        let pyramid = Vec::with_capacity(self.levels);
        for level in 0..self.levels {
            let scale = self.v_inv_scale_factor[level];
            let size = Size::new(
                cv_round_f32(image.cols() as f32 * scale),
                cv_round_f32(image.rows() as f32 * scale),
            );
            let whole_size = Size::new(
                size.width + EDGE_THRESHOLD * 2,
                size.height + EDGE_THRESHOLD * 2,
            );
            let temp = Mat::new_size_with_default(whole_size, image.typ(), Scalar::all(0.));
            // TODO: fill the pyramid and so on
        }
        self.image_pyramid = Some(pyramid);
    }

    pub fn get_levels(&self) -> usize {
        self.levels
    }
    pub fn get_scale_factor(&self) -> f64 {
        self.scale_factor
    }
    pub fn get_scale_factors(&self) -> &Vec<f32> {
        &self.v_scale_factor
    }
    pub fn get_inverse_scale_factors(&self) -> &Vec<f32> {
        &self.v_inv_scale_factor
    }
    pub fn get_scale_sigma2(&self) -> &Vec<f32> {
        &self.v_level_sigma2
    }
    pub fn get_inverse_scale_sigma2(&self) -> &Vec<f32> {
        &self.v_inv_level_sigma2
    }
}
