use nalgebra::{Isometry3, Matrix3, Matrix6, Matrix6x1, Rotation3, SMatrix, Vector3, Vector6};
use serde::{Deserialize, Serialize};

const EPS: f32 = 1e-4;

fn normalize_rotation(r: Matrix3<f32>) -> Matrix3<f32> {
    let svd = r.svd(true, true);
    let u = svd.u.expect("SVD U should be computed");
    let v_t = svd.v_t.expect("SVD V^T should be computed");
    u * v_t
}

pub fn right_jacobian_so3(v: Vector3<f32>) -> Matrix3<f32> {
    let d2 = v.norm_squared();
    let d = d2.sqrt();
    let w = v.cross_matrix();
    if d < EPS {
        Matrix3::identity()
    } else {
        Matrix3::identity() - w * ((1.0 - d.cos()) / d2) + w * w * ((d - d.sin()) / (d2 * d))
    }
}

pub fn right_jacobian_so3_xyz(x: f32, y: f32, z: f32) -> Matrix3<f32> {
    right_jacobian_so3(Vector3::new(x, y, z))
}

pub fn inverse_right_jacobian_so3(v: Vector3<f32>) -> Matrix3<f32> {
    let d2 = v.norm_squared();
    let d = d2.sqrt();
    let w = v.cross_matrix();
    if d < EPS {
        Matrix3::identity()
    } else {
        Matrix3::identity()
            + w * 0.5_f32
            + w * w * (1.0 / d2 - (1.0 + d.cos()) / (2.0 * d * d.sin()))
    }
}

pub fn inverse_right_jacobian_so3_xyz(x: f32, y: f32, z: f32) -> Matrix3<f32> {
    inverse_right_jacobian_so3(Vector3::new(x, y, z))
}

// IMU measurement (gyro, accelerometer and timestamp)
pub struct Point {
    pub a: Vector3<f32>,
    pub w: Vector3<f32>,
    pub t: f64,
}
impl Point {
    pub fn new(a: Vector3<f32>, w: Vector3<f32>, t: f64) -> Self {
        Point { a, w, t }
    }
    pub fn from_params(
        acc_x: f32,
        acc_y: f32,
        acc_z: f32,
        ang_vel_x: f32,
        ang_vel_y: f32,
        ang_vel_z: f32,
        t: f64,
    ) -> Self {
        Point {
            a: Vector3::new(acc_x, acc_y, acc_z),
            w: Vector3::new(ang_vel_x, ang_vel_y, ang_vel_z),
            t,
        }
    }
}

// IMU biases (gyro and accelerometer)
#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy)]
pub struct Bias {
    pub bax: f32,
    pub bay: f32,
    pub baz: f32,
    pub bwx: f32,
    pub bwy: f32,
    pub bwz: f32,
}
impl Bias {
    pub fn empty() -> Self {
        Bias {
            bax: 0.0,
            bay: 0.0,
            baz: 0.0,
            bwx: 0.0,
            bwy: 0.0,
            bwz: 0.0,
        }
    }
    pub fn from_params(bax: f32, bay: f32, baz: f32, bwx: f32, bwy: f32, bwz: f32) -> Self {
        Bias {
            bax,
            bay,
            baz,
            bwx,
            bwy,
            bwz,
        }
    }
}

// IMU calibration (Tbc, Tcb, noise)
#[derive(Serialize, Deserialize, Clone)]
pub struct Calib {
    pub tcb: Isometry3<f32>,
    pub tbc: Isometry3<f32>,
    cov: Vector6<f32>,
    cov_walk: Vector6<f32>,
    pub is_set: bool,
}
impl Calib {
    pub fn new() -> Self {
        Calib {
            tcb: Isometry3::identity(),
            tbc: Isometry3::identity(),
            cov: Vector6::zeros(),
            cov_walk: Vector6::zeros(),
            is_set: false,
        }
    }
    pub fn from_params(tbc: Isometry3<f32>, g: f32, a: f32, gw: f32, aw: f32) -> Self {
        let mut calib = Calib::new();
        calib.set(tbc, g, a, gw, aw);
        calib
    }
    pub fn set(&mut self, tbc: Isometry3<f32>, g: f32, a: f32, gw: f32, aw: f32) {
        self.is_set = true;
        self.tbc = tbc;
        self.tcb = self.tbc.inverse();
        let g2 = g * g;
        let a2 = a * a;
        let gw2 = gw * gw;
        let aw2 = aw * aw;
        self.cov = Vector6::new(g2, g2, g2, a2, a2, a2);
        self.cov_walk = Vector6::new(gw2, gw2, gw2, aw2, aw2, aw2);
    }
    pub fn set_cov(&mut self, cov: Vector6<f32>) {
        self.cov = cov;
    }
    pub fn set_cov_walk(&mut self, cov_walk: Vector6<f32>) {
        self.cov_walk = cov_walk;
    }

    pub fn get_cov(&self) -> Matrix6<f32> {
        Matrix6::from_diagonal(&self.cov)
    }
    pub fn get_cov_walk(&self) -> Matrix6<f32> {
        Matrix6::from_diagonal(&self.cov_walk)
    }
}

pub struct IntegratedRotation {
    pub delta_t: f32, // integration time
    pub delta_r: Matrix3<f32>,
    pub right_j: Matrix3<f32>, // right jacobian
}
impl IntegratedRotation {
    pub fn new() -> Self {
        IntegratedRotation {
            delta_t: 0.0,
            delta_r: Matrix3::identity(),
            right_j: Matrix3::identity(),
        }
    }
    pub fn from_params(ang_vel: &Vector3<f32>, imu_bias: &Bias, time: f32) -> Self {
        let x = ang_vel[0] - imu_bias.bwx * time;
        let y = ang_vel[1] - imu_bias.bwy * time;
        let z = ang_vel[2] - imu_bias.bwz * time;

        let d2 = x * x + y * y + z * z;
        let d = d2.sqrt();

        let v = Vector3::new(x, y, z);
        let w = v.cross_matrix();

        let delta_t = 0.0;
        let delta_r;
        let right_j;
        if d < EPS {
            delta_r = Matrix3::identity() + w;
            right_j = Matrix3::identity();
        } else {
            delta_r = Matrix3::identity() + w * (d.sin() / d) + w * w * ((1.0_f32 - d.cos()) / d2);
            right_j = Matrix3::identity() - w * ((1.0_f32 - d.cos()) / d2)
                + w * w * ((d - d.sin()) / (d2 * d));
        }

        IntegratedRotation {
            delta_t,
            delta_r,
            right_j,
        }
    }
}

// Preintegration of IMU measurements
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct Integrable {
    a: Vector3<f32>,
    w: Vector3<f32>,
    t: f32,
}
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Preintegrated {
    pub dt: f32,
    pub c: SMatrix<f32, 15, 15>,
    pub info: SMatrix<f32, 15, 15>,
    pub nga: Vector6<f32>,
    pub nga_walk: Vector6<f32>,
    pub b: Bias,
    pub dr: Matrix3<f32>,
    pub dv: Vector3<f32>,
    pub dp: Vector3<f32>,
    pub jrg: Matrix3<f32>,
    pub jvg: Matrix3<f32>,
    pub jva: Matrix3<f32>,
    pub jpg: Matrix3<f32>,
    pub jpa: Matrix3<f32>,
    pub avg_a: Vector3<f32>,
    pub avg_w: Vector3<f32>,
    // Updated bias
    bu: Bias,
    // Dif between original and updated bias, this is used to compute the updated values of preintegration
    db: Matrix6x1<f32>,
    measurements: Vec<Integrable>,
}
impl Preintegrated {
    pub fn from_bias_and_calib(b: &Bias, calib: &Calib) -> Self {
        let mut item = Self::default();
        item.nga = calib.cov;
        item.nga_walk = calib.cov_walk;
        item.initialize(b);
        item
    }
    pub fn initialize(&mut self, b: &Bias) {
        self.dr = Matrix3::identity();
        self.dv = Vector3::zeros();
        self.dp = Vector3::zeros();
        self.jrg = Matrix3::zeros();
        self.jvg = Matrix3::zeros();
        self.jva = Matrix3::zeros();
        self.jpg = Matrix3::zeros();
        self.jpa = Matrix3::zeros();
        self.c = SMatrix::zeros();
        self.info = SMatrix::zeros();
        self.db = Matrix6x1::zeros();
        self.b = *b;
        self.bu = *b;
        self.avg_a = Vector3::zeros();
        self.avg_w = Vector3::zeros();
        self.dt = 0.0;
        self.measurements.clear();
    }

    pub fn reintegrate(&mut self) {
        let aux = self.measurements.clone();
        let bu = self.bu; // Bias is Copy
        self.initialize(&bu);
        for m in &aux {
            self.integrate_new_measurement(m.a, m.w, m.t);
        }
    }

    pub fn integrate_new_measurement(&mut self, acc: Vector3<f32>, ang: Vector3<f32>, dt: f32) {
        self.measurements.push(Integrable {
            a: acc,
            w: ang,
            t: dt,
        });

        // Position is updated firstly, as it depends on previously computed velocity and rotations
        // Velocity is updated secondly, as it depends on previously computer rotation
        // Rotation is the last to be updated

        // Matrices to compute covariance
        let mut a: SMatrix<f32, 9, 9> = SMatrix::identity();
        let mut b: SMatrix<f32, 9, 6> = SMatrix::zeros();

        let acc: Vector3<f32> = Vector3::new(
            acc[0] - self.b.bax,
            acc[1] - self.b.bay,
            acc[2] - self.b.baz,
        );
        let acc_w: Vector3<f32> = Vector3::new(
            ang[0] - self.b.bwx,
            ang[1] - self.b.bwy,
            ang[2] - self.b.bwz,
        );

        self.avg_a = (self.dt * self.avg_a + self.dr * acc * dt) / (self.dt + dt);
        self.avg_w = (self.dt * self.avg_w + acc_w * dt) / (self.dt + dt);

        // Update delta position dp and velocity dv (rely on no-updated delta rotation)
        self.dp = self.dp + self.dv * dt + 0.5 * self.dr * acc * dt * dt;
        self.dv = self.dv + self.dr * acc * dt;

        // Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
        let w_acc = acc.cross_matrix();

        a.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(-self.dr * dt * w_acc));
        a.fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-0.5 * self.dr * dt * dt * w_acc));
        a.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(Matrix3::identity() * dt));
        b.fixed_view_mut::<3, 3>(3, 3).copy_from(&(self.dr * dt));
        b.fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(0.5_f32 * self.dr * dt * dt));

        // Update positions and velocity jacobians wrt bias correction
        self.jpa = self.jpa + self.jva * dt - 0.5 * self.dr * dt * dt;
        self.jpg = self.jpg + self.jvg * dt - 0.5 * self.dr * dt * dt * w_acc * self.jrg;
        self.jva = self.jva - self.dr * dt;
        self.jvg = self.jvg - self.dr * dt * w_acc * self.jrg;

        // Update delta rotation
        let dri = IntegratedRotation::from_params(&ang, &self.b, dt);
        self.dr = normalize_rotation(self.dr * dri.delta_r);

        // Compute rotation parts of matrices A and B
        a.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&dri.delta_r.transpose());
        b.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(dri.right_j * dt));

        // Update covariance
        // nga and nga_walk are stored as diagonal vectors; expand to 6×6 diagonal matrices
        // for the matrix products (mirrors Eigen::DiagonalMatrix<float,6> in the C++ original).
        let nga_mat = SMatrix::<f32, 6, 6>::from_diagonal(&self.nga);
        let nga_walk_mat = Matrix6::from_diagonal(&self.nga_walk);
        let c_sub = self.c.fixed_view::<9, 9>(0, 0).into_owned();
        self.c
            .fixed_view_mut::<9, 9>(0, 0)
            .copy_from(&(a * c_sub * a.transpose() + b * nga_mat * b.transpose()));
        let c_sub2 = self.c.fixed_view::<6, 6>(9, 9).into_owned();
        self.c
            .fixed_view_mut::<6, 6>(9, 9)
            .copy_from(&(c_sub2 + nga_walk_mat));

        // Update rotation jacobian wrt bias correction
        self.jrg = dri.delta_r.transpose() * self.jrg - dri.right_j * dt;

        // Total integrated time
        self.dt += dt;
    }

    pub fn merge_previous(&mut self, prev: &Preintegrated) {
        let bu = self.bu; // Bias is Copy
        let aux1 = prev.measurements.clone();
        let aux2 = self.measurements.clone();
        self.initialize(&bu);
        for m in &aux1 {
            self.integrate_new_measurement(m.a, m.w, m.t);
        }
        for m in &aux2 {
            self.integrate_new_measurement(m.a, m.w, m.t);
        }
    }

    pub fn set_new_bias(&mut self, b: &Bias) {
        self.bu = *b;
        self.db[0] = b.bwx - self.b.bwx;
        self.db[1] = b.bwy - self.b.bwy;
        self.db[2] = b.bwz - self.b.bwz;
        self.db[3] = b.bax - self.b.bax;
        self.db[4] = b.bay - self.b.bay;
        self.db[5] = b.baz - self.b.baz;
    }

    pub fn compute_bias_diff(&self, b: &Bias) -> Bias {
        Bias::from_params(
            b.bax - self.b.bax,
            b.bay - self.b.bay,
            b.baz - self.b.baz,
            b.bwx - self.b.bwx,
            b.bwy - self.b.bwy,
            b.bwz - self.b.bwz,
        )
    }

    pub fn get_delta_rotation(&self, b: &Bias) -> Matrix3<f32> {
        let dbg = Vector3::new(b.bwx - self.b.bwx, b.bwy - self.b.bwy, b.bwz - self.b.bwz);
        normalize_rotation(self.dr * Rotation3::new(self.jrg * dbg).into_inner())
    }

    pub fn get_delta_velocity(&self, b: &Bias) -> Vector3<f32> {
        let dbg = Vector3::new(b.bwx - self.b.bwx, b.bwy - self.b.bwy, b.bwz - self.b.bwz);
        let dba = Vector3::new(b.bax - self.b.bax, b.bay - self.b.bay, b.baz - self.b.baz);
        self.dv + self.jvg * dbg + self.jva * dba
    }

    pub fn get_delta_position(&self, b: &Bias) -> Vector3<f32> {
        let dbg = Vector3::new(b.bwx - self.b.bwx, b.bwy - self.b.bwy, b.bwz - self.b.bwz);
        let dba = Vector3::new(b.bax - self.b.bax, b.bay - self.b.bay, b.baz - self.b.baz);
        self.dp + self.jpg * dbg + self.jpa * dba
    }

    pub fn get_updated_delta_rotation(&self) -> Matrix3<f32> {
        let dbg = self.db.fixed_rows::<3>(0);
        normalize_rotation(self.dr * Rotation3::new(self.jrg * dbg).into_inner())
    }

    pub fn get_updated_delta_velocity(&self) -> Vector3<f32> {
        let dbg = self.db.fixed_rows::<3>(0);
        let dba = self.db.fixed_rows::<3>(3);
        self.dv + self.jvg * dbg + self.jva * dba
    }

    pub fn get_updated_delta_position(&self) -> Vector3<f32> {
        let dbg = self.db.fixed_rows::<3>(0);
        let dba = self.db.fixed_rows::<3>(3);
        self.dp + self.jpg * dbg + self.jpa * dba
    }

    pub fn get_original_delta_rotation(&self) -> Matrix3<f32> {
        self.dr
    }

    pub fn get_original_delta_velocity(&self) -> Vector3<f32> {
        self.dv
    }

    pub fn get_original_delta_position(&self) -> Vector3<f32> {
        self.dp
    }

    pub fn get_original_bias(&self) -> Bias {
        self.b
    }

    pub fn get_updated_bias(&self) -> Bias {
        self.bu
    }

    pub fn get_delta_bias(&self) -> Matrix6x1<f32> {
        self.db
    }
}
