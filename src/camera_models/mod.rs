use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::{Isometry3, Matrix2x1, Matrix2x3, Matrix3, Point2, Point3, Vector3};
use opencv::core::{KeyPoint, Mat, Point2f, Point3f};

use crate::two_view_reconstruction::ReconstructResult;

pub mod kannala_brandt8;
pub mod pinhole;

#[derive(Debug, Clone, Copy)]
enum Type {
    Pinhole,
    Fisheye,
}

static NEXT_GEOMETRIC_CAMERA_ID: AtomicU64 = AtomicU64::new(0);
fn next_geometric_camera_id() -> u64 {
    NEXT_GEOMETRIC_CAMERA_ID.fetch_add(1, Ordering::Relaxed)
}

// TODO: review if all this mix of datatypes is needed like this
// might be better to keep a more consistent formatting
// TODO: serialize
trait GeometricCamera {
    fn new() -> Self
    where
        Self: Sized;
    fn from_params(params: Vec<f32>) -> Self
    where
        Self: Sized;
    fn as_any(&self) -> &dyn Any;

    fn project(&self, p3d: &Point3f) -> Point2f;
    fn project_n(&self, p3d: &Point3<f32>) -> Point2<f32>;
    fn project_n_d(&self, p3d: &Point3<f64>) -> Point2<f64>;
    fn project_mat(&self, p3d: &Point3f) -> Point2<f32>;

    fn uncertainty(&self, p2d: &Matrix2x1<f64>) -> f32;

    fn unproject_n(&self, p2d: &Point2f) -> Point3<f32>;
    fn unproject(&self, p2d: &Point2f) -> Point3f;

    fn project_jac(&self, p3d: &Point3<f64>) -> Matrix2x3<f64>;

    fn reconstruct_with_two_views(
        &mut self,
        keys1: &Vec<KeyPoint>,
        keys2: &Vec<KeyPoint>,
        matches: &Vec<Option<usize>>,
        p3d: &Vec<Point3f>,
    ) -> Option<ReconstructResult>;

    fn to_k(&self) -> Mat;
    fn to_k_n(&self) -> Matrix3<f32>;

    fn epipolar_constrain(
        &self,
        other_camera: &dyn GeometricCamera,
        kp1: &KeyPoint,
        kp2: &KeyPoint,
        r12: &Matrix3<f32>,
        t12: &Point3<f32>,
        sigma_level: f32,
        unc: f32,
    ) -> bool;

    fn get_parameter(&self, i: usize) -> f32;
    fn set_parameter(&mut self, p: f32, i: usize);

    fn size(&self) -> usize;

    fn match_and_triangulate(
        &self,
        kp1: &KeyPoint,
        kp2: &KeyPoint,
        other: &dyn GeometricCamera,
        tcw1: &Isometry3<f32>,
        tcw2: &Isometry3<f32>,
        sigma_level1: f32,
        sigma_level2: f32,
        triangulated_3d: &Vector3<f32>,
    ) -> bool;

    fn get_id(&self) -> u64;
    fn get_type(&self) -> Type;
}
