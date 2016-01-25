use std::ops::{Add, Sub, Mul, Div};

#[macro_use]
extern crate nalgebra;
use nalgebra::ApproxEq;

extern crate num;
use num::{Float};

#[inline]
pub fn magic_number_1<T>() -> T
    where T: std::convert::From<f32>
{
    <T as std::convert::From<f32>>::from(700.)
}

#[inline]
pub fn magic_number_2<T>() -> T
    where T: std::convert::From<f32>
{
    <T as std::convert::From<f32>>::from(2595.)
}

pub fn hertz_from_mel<T>(mel: T) -> T
    where T: std::convert::From<f32> + Float + Div<T, Output=T> + Sub<T, Output=T>
{
    magic_number_1::<T>() *
        (<T as std::convert::From<f32>>::from(10.)
         .powf(mel / magic_number_2::<T>()) -
         <T as std::convert::From<f32>>::from(1.))
}

pub fn mel_from_hertz<T>(hertz: T) -> T
    where T: std::convert::From<f32> + Float + Mul<T, Output=T> + Add<T, Output=T>
{
    magic_number_2::<T>() *
        (<T as std::convert::From<f32>>::from(1.) +
         hertz / magic_number_1::<T>()).log10()
}

macro_rules! test_mel {
    ($float:ty) => {{
        assert_approx_eq_eps!(
            549.64 as $float, mel_from_hertz(440 as $float), 0.01);
        assert_approx_eq_eps!(
            440 as $float, hertz_from_mel(549.64 as $float), 0.01);

        let mel: $float = 0.;
        assert_approx_eq!(mel, mel_from_hertz(hertz_from_mel(mel)));
        let mel: $float = 100.;
        assert_approx_eq_eps!(
            mel, mel_from_hertz(hertz_from_mel(mel)), 0.0001);

        let hertz: $float = 0.;
        assert_approx_eq!(hertz, hertz_from_mel(mel_from_hertz(hertz)));
        let hertz: $float = 44100. / 2.;
        assert_approx_eq!(hertz, hertz_from_mel(mel_from_hertz(hertz)));
    }}
}

#[test]
fn test_mel_f32() {
    test_mel!(f32);
}

#[test]
fn test_mel_f64() {
    test_mel!(f64);
}
