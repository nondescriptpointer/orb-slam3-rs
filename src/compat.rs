#[inline]
pub fn cv_round_f32(x: f32) -> i32 {
    x.round_ties_even() as i32
}

#[inline]
pub fn cv_round_f64(x: f64) -> i32 {
    x.round_ties_even() as i32
}
