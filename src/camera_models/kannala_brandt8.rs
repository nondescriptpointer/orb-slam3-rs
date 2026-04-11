use nalgebra::{
    Isometry3, Matrix, Matrix2x3, Matrix3, Matrix3x4, Matrix4, Point2, Point3, Vector3,
};
use opencv::{
    calib3d,
    core::{
        CV_32F, KeyPoint, KeyPointTrait, KeyPointTraitConst, Mat, MatExprTraitConst, Point2f,
        Point3f, TermCriteria, TermCriteria_Type, Vector,
    },
};

use crate::{
    camera_models::{GeometricCamera, Type, next_geometric_camera_id},
    two_view_reconstruction::{ReconstructResult, TwoViewReconstruction},
};

struct KannalaBrandt8 {
    parameters: Vec<f32>,
    id: u64,
    camera_type: Type,
    tvr: Option<TwoViewReconstruction>,
    precision: f32,
    lapping_area: Vec<usize>,
}

struct TriangulateMatchesResult {
    z1: f32,
    p3d: Vector3<f32>,
}
enum TriangulateMatchesError {
    Parallax,    // -1
    Z1,          // -2
    Z2,          // -3
    Projection1, // -4,
    Projection2, // -5,
    Triangulate, // -6 (not in original codebase)
}

fn triangulate(
    p1: Point2f,
    p2: Point2f,
    tcw1: Matrix3x4<f32>,
    tcw2: Matrix3x4<f32>,
) -> Option<Vector3<f32>> {
    let mut a = Matrix4::<f32>::zeros();
    a.row_mut(0).copy_from(&(p1.x * tcw1.row(2) - tcw1.row(0)));
    a.row_mut(1).copy_from(&(p1.y * tcw1.row(2) - tcw1.row(1)));
    a.row_mut(2).copy_from(&(p2.x * tcw2.row(2) - tcw2.row(0)));
    a.row_mut(3).copy_from(&(p2.y * tcw2.row(2) - tcw2.row(1)));

    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let x3d_h = v_t.row(3).transpose();

    let w = x3d_h[3];
    if w.abs() < f32::EPSILON {
        return None;
    }

    Some(Vector3::new(x3d_h[0] / w, x3d_h[1] / w, x3d_h[2] / w))
}

fn isometry_to_matrix3x4(tcw: &Isometry3<f32>) -> Matrix3x4<f32> {
    let r = tcw.rotation.to_rotation_matrix().into_inner();
    let t = tcw.translation.vector;

    Matrix3x4::new(
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        t[0],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        t[1],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        t[2],
    )
}

impl KannalaBrandt8 {
    fn project_xyz(&self, x: f32, y: f32, z: f32) -> (f32, f32) {
        let x2_plus_y2 = x * x + y * y;
        let theta = x2_plus_y2.sqrt().atan2(z);
        let psi = y.atan2(x);

        let theta2 = theta * theta;
        let theta3 = theta * theta2;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let r = theta
            + self.parameters[4] * theta3
            + self.parameters[5] * theta5
            + self.parameters[6] * theta7
            + self.parameters[7] * theta9;

        (
            self.parameters[0] * r * psi.cos() + self.parameters[2],
            self.parameters[1] * r * psi.sin() + self.parameters[3],
        )
    }

    fn with_params_and_precision(params: Vec<f32>, precision: f32) -> Self {
        if params.len() != 8 {
            panic!("Invalid GeometricCamera params size");
        }
        KannalaBrandt8 {
            parameters: params,
            id: next_geometric_camera_id(),
            camera_type: Type::Fisheye,
            tvr: None,
            precision: precision,
            lapping_area: vec![2, 0],
        }
    }

    fn triangulate_matches(
        &self,
        camera: &dyn GeometricCamera,
        kp1: &KeyPoint,
        kp2: &KeyPoint,
        r12: &Matrix3<f32>,
        t12: &Vector3<f32>,
        sigma_level: f32,
        unc: f32,
    ) -> Result<TriangulateMatchesResult, TriangulateMatchesError> {
        // TODO: return result instead of option?
        let r1 = self.unproject_n(&kp1.pt());
        let r2 = camera.unproject_n(&kp2.pt());

        // Check parallax
        let r21 = r12 * r2;
        let cos_parallax_rays = r1.dot(&r21) / (r1.norm() * r21.norm());
        if cos_parallax_rays > 0.9998 {
            return Err(TriangulateMatchesError::Parallax); // -1
        }

        // Parallax is good, so we try to triangulate
        let p11 = Point2f::new(r1[0], r1[1]);
        let p22 = Point2f::new(r2[0], r2[1]);

        let tcw1 =
            Matrix3x4::<f32>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let r21 = r12.transpose();
        let t2: Vector3<f32> = -r21 * t12;
        let tcw2: Matrix3x4<f32> = Matrix3x4::new(
            r21[(0, 0)],
            r21[(0, 1)],
            r21[(0, 2)],
            t2[0],
            r21[(1, 0)],
            r21[(1, 1)],
            r21[(1, 2)],
            t2[1],
            r21[(2, 0)],
            r21[(2, 1)],
            r21[(2, 2)],
            t2[2],
        );

        let x3d: Vector3<f32> =
            triangulate(p11, p22, tcw1, tcw2).ok_or(TriangulateMatchesError::Triangulate)?;

        let z1 = x3d[2];
        if z1 <= 0. {
            return Err(TriangulateMatchesError::Z1); // -2
        }
        let x3d_cam2 = r21 * x3d + t2;
        let z2 = x3d_cam2[2];
        if z2 <= 0. {
            return Err(TriangulateMatchesError::Z2); // -3
        }

        // Check reprojection error
        let uv1 = self.project_n(&x3d.into());
        let errx1 = uv1.x - kp1.pt().x;
        let erry1 = uv1.y - kp1.pt().y;
        if (errx1 * errx1 + erry1 * erry1) > 5.991 * sigma_level {
            return Err(TriangulateMatchesError::Projection1); // -4
        }

        let t2 = Vector3::new(tcw2[(0, 3)], tcw2[(1, 3)], tcw2[(2, 3)]);
        let x3d2 = r21 * x3d + t2;
        let uv2 = camera.project_n(&x3d2.into());
        let errx2 = uv2.x - kp2.pt().x;
        let erry2 = uv2.y - kp2.pt().y;
        if (errx2 * errx2 + erry2 * erry2) > 5.991 * unc {
            return Err(TriangulateMatchesError::Projection2); // -5
        }

        Ok(TriangulateMatchesResult { z1, p3d: x3d })
    }

    fn is_equal(&self, other: &dyn GeometricCamera) -> bool {
        let Some(other) = other.as_any().downcast_ref::<KannalaBrandt8>() else {
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
        if (self.precision - other.precision).abs() > 1e-6 {
            is_same = false;
        }
        is_same
    }
}

impl GeometricCamera for KannalaBrandt8 {
    fn new() -> Self
    where
        Self: Sized,
    {
        KannalaBrandt8 {
            parameters: Vec::with_capacity(8),
            id: next_geometric_camera_id(),
            camera_type: Type::Fisheye,
            tvr: None,
            precision: 1e-6,
            lapping_area: vec![2, 0],
        }
    }
    fn with_params(params: Vec<f32>) -> Self
    where
        Self: Sized,
    {
        if params.len() != 8 {
            panic!("Invalid GeometricCamera params size");
        }
        KannalaBrandt8 {
            parameters: params,
            id: next_geometric_camera_id(),
            camera_type: Type::Fisheye,
            tvr: None,
            precision: 1e-6,
            lapping_area: vec![2, 0],
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn project(&self, p3d: &Point3f) -> Point2f {
        let (u, v) = self.project_xyz(p3d.x, p3d.y, p3d.z);
        Point2f::new(u, v)
    }
    fn project_n(&self, p3d: &Point3<f32>) -> Point2<f32> {
        let (u, v) = self.project_xyz(p3d.x, p3d.y, p3d.z);
        Point2::new(u, v)
    }
    fn project_n_d(&self, p3d: &Point3<f64>) -> Point2<f64> {
        let x2_plus_y2 = p3d.x * p3d.x + p3d.y * p3d.y;
        let theta = x2_plus_y2.sqrt().atan2(p3d.z);
        let psi = p3d.y.atan2(p3d.x);

        let theta2 = theta * theta;
        let theta3 = theta * theta2;
        let theta5 = theta3 * theta2;
        let theta7 = theta5 * theta2;
        let theta9 = theta7 * theta2;

        let r = theta
            + self.parameters[4] as f64 * theta3
            + self.parameters[5] as f64 * theta5
            + self.parameters[6] as f64 * theta7
            + self.parameters[7] as f64 * theta9;

        Point2::new(
            self.parameters[0] as f64 * r * psi.cos() + self.parameters[2] as f64,
            self.parameters[1] as f64 * r * psi.sin() + self.parameters[3] as f64,
        )
    }
    fn project_mat(&self, p3d: &Point3f) -> Point2<f32> {
        let (u, v) = self.project_xyz(p3d.x, p3d.y, p3d.z);
        Point2::new(u, v)
    }

    fn uncertainty(&self, _p2d: &nalgebra::Matrix2x1<f64>) -> f32 {
        return 1.0;
    }

    fn unproject(&self, p2d: &Point2f) -> Point3f {
        // Use Newthon method to solve for theta with good precision (err ~ e-6)
        let pw = Point2f::new(
            (p2d.x - self.parameters[2]) / self.parameters[0],
            (p2d.y - self.parameters[3]) / self.parameters[1],
        );
        let mut scale = 1.0f32;
        let theta_d = (pw.x * pw.x + pw.y * pw.y)
            .sqrt()
            .clamp(-std::f32::consts::PI / 2.0, std::f32::consts::PI / 2.0);

        if theta_d > 1e-8 {
            // Compensate distortion iteratively
            let mut theta = theta_d;

            for _ in 0..10 {
                let theta2 = theta * theta;
                let theta4 = theta2 * theta2;
                let theta6 = theta4 * theta2;
                let theta8 = theta4 * theta4;

                let t2 = self.parameters[4] * theta2;
                let t4 = self.parameters[5] * theta4;
                let t6 = self.parameters[6] * theta6;
                let t8 = self.parameters[7] * theta8;

                let theta_fix = (theta * (1.0 + t2 + t4 + t6 + t8) - theta_d)
                    / (1.0 + 3.0 * t2 + 5.0 * t4 + 7.0 * t6 + 9.0 * t8);

                theta -= theta_fix;

                if theta_fix.abs() < self.precision {
                    break;
                }
            }
            scale = theta.tan() / theta_d;
        }

        Point3f::new(pw.x * scale, pw.y * scale, 1.)
    }
    fn unproject_n(&self, p2d: &Point2f) -> Vector3<f32> {
        let ray = self.unproject(p2d);
        Vector3::new(ray.x, ray.y, ray.z)
    }

    fn project_jac(&self, p3d: &Point3<f64>) -> nalgebra::Matrix2x3<f64> {
        let x = p3d[0];
        let y = p3d[1];
        let z = p3d[2];
        let x2 = x * x;
        let y2 = y * y;
        let z2 = z * z;
        let r2 = x2 + y2;
        let r = r2.sqrt();
        let r3 = r2 * r;
        let theta = r.atan2(z);

        let theta2 = theta * theta;
        let theta3 = theta * theta2;
        let theta4 = theta2 * theta2;
        let theta5 = theta3 * theta2;
        let theta6 = theta2 * theta4;
        let theta7 = theta6 * theta;
        let theta8 = theta7 * theta;
        let theta9 = theta7 * theta2;

        let fx = self.parameters[0] as f64;
        let fy = self.parameters[1] as f64;
        let k0 = self.parameters[4] as f64;
        let k1 = self.parameters[5] as f64;
        let k2 = self.parameters[6] as f64;
        let k3 = self.parameters[7] as f64;

        let f = theta + theta3 * k0 + theta5 * k1 + theta7 * k2 + theta9 * k3;
        let fd =
            1.0 + 3.0 * k0 * theta2 + 5.0 * k1 * theta4 + 7.0 * k2 * theta6 + 9.0 * k3 * theta8;

        let mut jac = Matrix2x3::zeros();

        jac[(0, 0)] = fx * (fd * z * x2 / (r2 * (r2 + z2)) + f * y2 / r3);
        jac[(1, 0)] = fy * (fd * z * y * x / (r2 * (r2 + z2)) - f * y * x / r3);

        jac[(0, 1)] = fx * (fd * z * y * x / (r2 * (r2 + z2)) - f * y * x / r3);
        jac[(1, 1)] = fy * (fd * z * y2 / (r2 * (r2 + z2)) + f * x2 / r3);

        jac[(0, 2)] = -fx * fd * x / (r2 + z2);
        jac[(1, 2)] = -fy * fd * y / (r2 + z2);

        jac
    }

    fn reconstruct_with_two_views(
        &mut self,
        keys1: &Vec<opencv::core::KeyPoint>,
        keys2: &Vec<opencv::core::KeyPoint>,
        matches: &Vec<Option<usize>>,
    ) -> Option<ReconstructResult> {
        if self.tvr.is_none() {
            let k = self.to_k_n();
            self.tvr = Some(TwoViewReconstruction::new(k));
        }

        // Correct FishEye distortion
        let mut pts1 = Vector::<Point2f>::new();
        let mut pts2 = Vector::<Point2f>::new();
        for kp in keys1 {
            pts1.push(kp.pt());
        }
        for kp in keys2 {
            pts2.push(kp.pt());
        }

        let d = Mat::from_slice_2d(&[
            &[self.parameters[4]],
            &[self.parameters[5]],
            &[self.parameters[6]],
            &[self.parameters[7]],
        ])
        .ok()?;
        let r = Mat::eye(3, 3, CV_32F).ok()?.to_mat().ok()?;
        let k = self.to_k();

        let mut pts1_ud = Vector::<Point2f>::new();
        let mut pts2_ud = Vector::<Point2f>::new();

        // reasonable default criteria
        let criteria = TermCriteria::new(
            (TermCriteria_Type::COUNT as i32) | (TermCriteria_Type::EPS as i32),
            10,
            1e-8,
        )
        .ok()?;

        calib3d::fisheye_undistort_points(&pts1, &mut pts1_ud, &k, &d, &r, &k, criteria).ok()?;
        calib3d::fisheye_undistort_points(&pts2, &mut pts2_ud, &k, &d, &r, &k, criteria).ok()?;

        let mut keys_un1 = keys1.clone();
        let mut keys_un2 = keys2.clone();
        for (i, kp) in keys_un1.iter_mut().enumerate() {
            kp.set_pt(pts1_ud.get(i).ok()?);
        }
        for (i, kp) in keys_un2.iter_mut().enumerate() {
            kp.set_pt(pts2_ud.get(i).ok()?);
        }

        if let Some(tvr) = &mut self.tvr {
            tvr.reconstruct(&keys_un1, &keys_un2, matches)
        } else {
            None
        }
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
        sigma_level: f32,
        unc: f32,
    ) -> bool {
        let result = self.triangulate_matches(
            other_camera,
            kp1,
            kp2,
            r12,
            &nalgebra::Vector3::new(t12.x, t12.y, t12.z),
            sigma_level,
            unc,
        );

        if let Ok(result) = result {
            result.z1 > 0.0001
        } else {
            false
        }
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
        kp1: &opencv::core::KeyPoint,
        kp2: &opencv::core::KeyPoint,
        other: &dyn GeometricCamera,
        tcw1: &nalgebra::Isometry3<f32>,
        tcw2: &nalgebra::Isometry3<f32>,
        sigma_level1: f32,
        sigma_level2: f32,
    ) -> Option<Vector3<f32>> {
        let mat_tcw1 = isometry_to_matrix3x4(tcw1);
        let rcw1: Matrix3<f32> = tcw1.rotation.to_rotation_matrix().into_inner();
        let rwc1 = rcw1.transpose();

        let mat_tcw2 = isometry_to_matrix3x4(tcw2);
        let rcw2: Matrix3<f32> = tcw2.rotation.to_rotation_matrix().into_inner();
        let rwc2 = rcw2.transpose();

        let ray1c = self.unproject(&kp1.pt());
        let ray2c = other.unproject(&kp2.pt());

        let r1 = Vector3::new(ray1c.x, ray1c.y, ray1c.z);
        let r2 = Vector3::new(ray2c.x, ray2c.y, ray2c.z);

        // Check parallax between rays
        let ray1 = rwc1 * r1;
        let ray2 = rwc2 * r2;
        let cos_parallax_rays = ray1.dot(&ray2) / (ray1.norm() * ray2.norm());

        // If parallax is lower than 0.9998, reject the match
        if cos_parallax_rays > 0.9998 {
            return None;
        }

        // Parallax is good, so we try to triangulate
        let p11 = Point2f::new(ray1c.x, ray1c.y);
        let p22 = Point2f::new(ray2c.x, ray2c.y);

        let x3d = triangulate(p11, p22, mat_tcw1, mat_tcw2)?;

        // Check triangulation in front of camera
        let z1 = rcw1.row(2).transpose().dot(&x3d) + tcw1.translation.z;
        if z1 <= 0. {
            return None;
        }
        let z2 = rcw2.row(2).transpose().dot(&x3d) + tcw2.translation.z;
        if z2 <= 0. {
            return None;
        }

        // Check projection error in first keyframe
        // Transform point into camera reference system
        let x3d1 = rcw1 * x3d + tcw1.translation.vector;
        let uv1 = self.project_n(&Point3::from(x3d1));
        let errx1 = uv1.x - kp1.pt().x;
        let erry1 = uv1.y - kp1.pt().y;
        if (errx1 * errx1 + erry1 * erry1) > 5.991 * sigma_level1 {
            //Reprojection error is high
            return None;
        }
        // Check reprojection error in second keyframe
        // Transform point into camera reference system
        let x3d2 = rcw2 * x3d + tcw2.translation.vector;
        let uv2 = other.project_n(&Point3::from(x3d2));
        let errx2 = uv2.x - kp2.pt().x;
        let erry2 = uv2.y - kp2.pt().y;
        if (errx2 * errx2 + erry2 * erry2) > 5.991 * sigma_level2 {
            //Reprojection error is high
            return None;
        }

        Some(x3d)
    }

    fn get_id(&self) -> u64 {
        self.id
    }

    fn get_type(&self) -> Type {
        self.camera_type
    }
}
