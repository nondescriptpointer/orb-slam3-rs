use crate::camera_models::{GeometricCamera, Type, next_geometric_camera_id};

use nalgebra::{Matrix2x3, Matrix3, Point2, Point3};
use opencv::core::{KeyPointTraitConst, Mat, Point2f, Point3f};
use sophus::autodiff::linalg::VecF64;
use sophus::autodiff::prelude::IsVector;
use sophus::lie::Rotation3F64;

struct Pinhole {
    parameters: Vec<f32>,
    id: u64,
    camera_type: Type,
    //two_view_reconstruction: TODO
}

impl GeometricCamera for Pinhole {
    fn new() -> Self
    where
        Self: Sized,
    {
        Pinhole {
            parameters: Vec::with_capacity(4),
            id: next_geometric_camera_id(),
            camera_type: Type::Pinhole,
        }
    }
    fn from_params(params: Vec<f32>) -> Self
    where
        Self: Sized,
    {
        if params.len() != 4 {
            panic!("Invalid GeometricCamera params size");
        }
        Pinhole {
            parameters: params,
            id: next_geometric_camera_id(),
            camera_type: Type::Pinhole,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn project(&self, p3d: &Point3f) -> Point2f {
        Point2f::new(
            self.parameters[0] * p3d.x / p3d.z + self.parameters[2],
            self.parameters[1] * p3d.y / p3d.z + self.parameters[3],
        )
    }
    fn project_n(&self, p3d: &Point3<f32>) -> Point2<f32> {
        Point2::new(
            self.parameters[0] * p3d[0] / p3d[2] + self.parameters[2],
            self.parameters[1] * p3d[1] / p3d[2] + self.parameters[3],
        )
    }
    fn project_n_d(&self, p3d: &Point3<f64>) -> Point2<f64> {
        Point2::new(
            self.parameters[0] as f64 * p3d[0] / p3d[2] + self.parameters[2] as f64,
            self.parameters[1] as f64 * p3d[1] / p3d[2] + self.parameters[3] as f64,
        )
    }
    fn project_mat(&self, p3d: &Point3f) -> Point2<f32> {
        let point = self.project(p3d);
        Point2::new(point.x, point.y)
    }

    fn uncertainty(&self, _p2d: &nalgebra::Matrix2x1<f64>) -> f32 {
        return 1.0;
    }

    fn unproject(&self, p2d: &Point2f) -> Point3f {
        Point3f::new(
            (p2d.x - self.parameters[2]) / self.parameters[0],
            (p2d.y - self.parameters[3]) / self.parameters[1],
            1.,
        )
    }
    fn unproject_n(&self, p2d: &Point2f) -> Point3<f32> {
        Point3::new(
            (p2d.x - self.parameters[2]) / self.parameters[0],
            (p2d.y - self.parameters[3]) / self.parameters[1],
            1.,
        )
    }

    fn project_jac(&self, p3d: &Point3<f64>) -> nalgebra::Matrix2x3<f64> {
        Matrix2x3::new(
            self.parameters[0] as f64 / p3d[2],
            0.,
            -self.parameters[0] as f64 * p3d[0] / (p3d[2] * p3d[2]),
            0.,
            self.parameters[1] as f64 / p3d[2],
            -self.parameters[1] as f64 * p3d[1] / (p3d[2] * p3d[2]),
        )
    }

    fn reconstruct_with_two_views(
        &self,
        keys1: &Vec<opencv::core::KeyPoint>,
        keys2: &Vec<opencv::core::KeyPoint>,
        matches: &Vec<usize>,
        t21: &nalgebra::Isometry3<f32>,
        p3d: &Vec<Point3f>,
        triangulated: &Vec<bool>,
    ) -> bool {
        // TODO
        true
    }

    fn to_k(&self) -> Mat {
        let row0 = [self.parameters[0], 0., self.parameters[2]];
        let row1 = [0., self.parameters[1], self.parameters[3]];
        let row2 = [0., 0., 1.];
        Mat::from_slice_2d(&[&row0, &row1, &row2]).expect("Failed to create matrix")
    }
    fn to_k_n(&self) -> nalgebra::Matrix3<f32> {
        Matrix3::new(
            self.parameters[0],
            0.,
            self.parameters[2],
            0.,
            self.parameters[1],
            self.parameters[3],
            0.,
            0.,
            1.,
        )
    }

    fn epipolar_constrain(
        &self,
        other_camera: &dyn GeometricCamera,
        kp1: &opencv::core::KeyPoint,
        kp2: &opencv::core::KeyPoint,
        r12: &Matrix3<f32>,
        t12: &Point3<f32>,
        _sigma_level: f32,
        unc: f32,
    ) -> bool {
        // Compute fundamental Matrix
        // `Rotation3::hat` is only available for `f64` in sophus (`IsScalar`); promote, then cast back.
        // TODO: optimize
        let omega = VecF64::<3>::from_array([t12[0] as f64, t12[1] as f64, t12[2] as f64]);
        let t12x = Rotation3F64::hat(omega).map(|v| v as f32);
        let k1 = self.to_k_n();
        let k2 = other_camera.to_k_n();
        let f12 = k1.transpose().try_inverse().unwrap() * t12x * r12 * k2.try_inverse().unwrap();

        // Epipolar line in second line
        let a = kp1.pt().x * f12[(0, 0)] + kp1.pt().y * f12[(1, 0)] + f12[(2, 0)];
        let b = kp1.pt().x * f12[(0, 1)] + kp1.pt().y * f12[(1, 1)] + f12[(2, 1)];
        let c = kp1.pt().x * f12[(0, 2)] + kp1.pt().y * f12[(1, 2)] + f12[(2, 2)];

        let num = a * kp2.pt().x + b * kp2.pt().y + c;
        let den = a * a + b * b;
        if den == 0. {
            return false;
        }
        let dsqr = num * num / den;

        dsqr < 3.84 * unc
    }

    fn get_parameter(&self, i: usize) -> f32 {
        *self.parameters.get(i).expect("Unknwown param")
    }
    fn set_parameter(&mut self, p: f32, i: usize) {
        self.parameters[i] = p;
    }

    fn size(&self) -> usize {
        self.parameters.len()
    }

    fn match_and_triangulate(
        &self,
        _kp1: &opencv::core::KeyPoint,
        _kp2: &opencv::core::KeyPoint,
        _other: &dyn GeometricCamera,
        _tcw1: &nalgebra::Isometry3<f32>,
        _tcw2: &nalgebra::Isometry3<f32>,
        _sigma_level1: f32,
        _sigma_level2: f32,
        _triangulated_3d: &nalgebra::Vector3<f32>,
    ) -> bool {
        false
    }

    fn get_id(&self) -> u64 {
        self.id
    }

    fn get_type(&self) -> Type {
        self.camera_type
    }
}

impl Pinhole {
    fn from_pinhole(other: &Self) -> Self {
        Pinhole {
            parameters: other.parameters.clone(),
            id: next_geometric_camera_id(),
            camera_type: Type::Pinhole,
        }
    }

    fn is_equal(&self, other: &dyn GeometricCamera) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Pinhole>() else {
            return false;
        };
        if self.size() != other.size() {
            return false;
        }
        let mut is_same = true;
        for i in 0..self.size() {
            if (self.parameters[i] - other.parameters[i]).abs() > 1e-6 {
                is_same = false;
                break;
            }
        }
        is_same
    }
}
