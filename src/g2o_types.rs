use nalgebra::{Matrix3, SMatrix, SymmetricEigen, Vector3};

type Matrix15d = SMatrix<f64, 15, 15>;

#[derive(Clone)]
pub struct ConstraintPoseIMU {
    rwb: Matrix3<f64>,
    twb: Vector3<f64>,
    vwb: Vector3<f64>,
    bg: Vector3<f64>,
    ba: Vector3<f64>,
    h: Matrix15d,
}
impl ConstraintPoseIMU {
    pub fn new(
        rwb: Matrix3<f64>,
        twb: Vector3<f64>,
        vwb: Vector3<f64>,
        bg: Vector3<f64>,
        ba: Vector3<f64>,
        h: SMatrix<f64, 15, 15>,
    ) -> Self {
        let h = (h + h) / 2.;
        let es = SymmetricEigen::new(h);
        let mut eigs = es.eigenvalues;
        for i in 0..15 {
            if eigs[i] < 1e-12 {
                eigs[i] = 0.;
            }
        }
        let h = es.eigenvectors * Matrix15d::from_diagonal(&eigs) * es.eigenvectors.transpose();

        ConstraintPoseIMU {
            rwb,
            twb,
            vwb,
            bg,
            ba,
            h,
        }
    }
}
