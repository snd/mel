/*!
```
extern crate hertz;

fn main() {
    let sample_rate = 44100;
    let window_size = 4096;
    let power_spectrum_size = window_size / 2;
    let filter_count = 100;

    for (row, col, value) in enumerate_mel_scaling_matrix(
        sample_rate as f64,
        window_size,
        power_spectrum_size,
        filter_count,
    ) {

    }
}
```
*/

use std::ops::{Add, Sub, Mul, Div, IndexMut};

#[macro_use]
extern crate nalgebra;
use nalgebra::ApproxEq;

extern crate num;
use num::{Float, Zero, NumCast, ToPrimitive};

extern crate hertz;
extern crate apodize;

extern crate itertools;
use itertools::linspace;

macro_rules! from {
    ($expr:expr, $src:ty, $dst:ty) => {
        <$dst as std::convert::From<$src>>::from($expr)
    }
}

#[inline]
pub fn hertz_from_mel<T>(mel: T) -> T
    where T: std::convert::From<f32> + Float + Div<T, Output=T> + Sub<T, Output=T>
{
    from!(700., f32, T) *
        (from!(10., f32, T).powf(
            mel / from!(2595., f32, T)) - from!(1., f32, T))
}

#[inline]
pub fn mel_from_hertz<T>(hertz: T) -> T
    where T: std::convert::From<f32> + Float + Mul<T, Output=T> + Add<T, Output=T>
{
    from!(2595., f32, T) *
        (from!(1., f32, T) + hertz / from!(700., f32, T)).log10()
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
        let mel: $float = 3000.;
        assert_approx_eq_eps!(
            mel, mel_from_hertz(hertz_from_mel(mel)), 0.0001);

        let hertz: $float = 0.;
        assert_approx_eq!(hertz, hertz_from_mel(mel_from_hertz(hertz)));
        let hertz: $float = 1000.;
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

/// col index changes fastest
struct MelScalingMatrixEnumerator<T, WindowIter> {
    // parameters

    /// equivalent to col count
    input_size: usize,
    /// equivalent to row count
    output_size: usize,

    mel_from_hertz: fn(T) -> T,
    hertz_from_mel: fn(T) -> T,

    window_function: fn(usize) -> WindowIter,

    // state

    row_index: usize,
    col_index: usize,

    start_mels_iter: itertools::Linspace<T>,
    end_mels_iter: itertools::Linspace<T>,

    // this gets set anew for every row
    window_size: usize,
    window_iter: WindowIter
}

impl<T, WindowIter> MelScalingMatrixEnumerator<T, WindowIter>
    where T: Zero
{
    #[inline]
    pub fn is_done(&self) -> bool {
        self.is_at_end_of_row() && self.is_at_end_of_col()
    }

    #[inline]
    pub fn is_at_end_of_col(&self) -> bool {
        self.output_size <= self.row_index
    }

    #[inline]
    pub fn is_at_end_of_row(&self) -> bool {
        self.input_size <= self.col_index
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.row_index * self.input_size + self.col_index
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        // TODO is this off by one ?
        self.len() - self.index()
    }
}

impl<T, WindowIter> Iterator for MelScalingMatrixEnumerator<T, WindowIter>
    where T: Zero
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.is_done() {
            return None;
        }
        Some(T::zero())
        //     Some(value) => {
        //         if self.is_at_end_of_row() {
        //             self.col_index = 0;
        //             self.row_index += 1;
        //         } else {
        //             self.col_index += 1;
        //         }
        //         value
        //     },
        // }
    }

        // // iterate row in outer loop (slow)
        // // iterate col in inner loop (fast)
        // // fill matrix with zeros
        // for i_row in 0..output_size {
        //     for i_col in 0..input_size {
        //         matrix[(i_row, i_col)] = T::zero();
        //     }
        // }
        //
        //
        // for (i_mel, (start_mel, end_mel)) in start_mels_iter
        //     .zip(end_mels_iter)
        //     .enumerate()
        // {
        //     let start_hertz = hertz_from_mel(start_mel) as usize;
        //     let end_hertz = hertz_from_mel(end_mel) as usize;
        //     assert!(start_hertz < end_hertz);
        //     let window_size = end_hertz - start_hertz;
        //     assert!(0 < window_size);
        //     let window_iter = window_function(window_size);
        //     for (ifreq, factor) in window_iter.enumerate() {
        //         matrix[(imel, ifreq)] = factor / window_size;
        //     }
        // }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl<T, WindowIter> ExactSizeIterator for MelScalingMatrixEnumerator<T, WindowIter>
    where T: Zero
{
    #[inline]
    fn len(&self) -> usize {
        self.input_size * self.output_size
    }
}

// TODO better name
pub fn mel_scaling_matrix_indexes<T> (
    mels: Range<T>,
) -> Range<usize>
{
    let start_hertz = hertz_from_mel(mels.start);
    let end_hertz = hertz_from_mel(mels.end);
    assert!(start_hertz < end_hertz);

    // divide by max_hertz
    // multiply with output_size

    let start_index = <usize as NumCast>::from()
        .expect("type `usize` can't represent a certain value of type of `T`.
                choose a `T` that can represent fewer numbers.");
    let end_index = <usize as NumCast>::from()
        .expect("type `usize` can't represent a certain value of type of `T`.
                choose a `T` that can represent fewer numbers.");

    start_index..end_index
}

/// (or many vectors at once in form of a matrix)
/// to transform that vector into mel space.
/// efficiently
/// rows are mel
/// cols are freq
/// TODO get frequency index pair for nth mel
/// TODO enumerate scaling vector
/// TODO which dimension first
pub fn enumerate_mel_scaling_matrix_base<T, WindowIter>(
    // TODO more descriptive argument names
    min_hertz: T,
    max_hertz: T,
    // TODO shape
    input_size: usize,
    output_size: usize,
    mel_from_hertz: fn(T) -> T,
    hertz_from_mel: fn(T) -> T,
    window_function: fn(usize) -> WindowIter
) -> MelScalingMatrixEnumerator<T, WindowIter>
    where WindowIter: Iterator<Item=T>,
          T: std::cmp::PartialOrd + Float + ToPrimitive + FromPrimitive
          usize: itertools::misc::ToFloat<T>
{
    assert!(min_freq < max_freq);
    assert!(output_size < input_size);
    assert!(0 < output_size);
    assert!(0 < input_size);

    let min_mel = mel_from_hertz(min_hertz);
    let max_mel = mel_from_hertz(max_hertz);

    let mut start_mels_iter = linspace(min_mel, max_mel, output_size);
    let mut end_mels_iter = linspace(max_mel, max_mel, output_size);
    end_mels_iter.next();

    let first_window_start_mel = start_mels_iter.next().unwrap();
    let first_window_end_mel= end_mels_iter.next().unwrap();
    assert!(first_window_start_mel < first_window_end_mel);
    let first_indexes = mel_scaling_matrix_indexes(
        first_window_start_mel..first_window_end_mel);


    let first_window_size = first_window_end_hertz - first_window_start_hertz;

    let first_window_iter = window_function(first_window_size);

    MelScalingMatrixEnumerator::<T, WindowIter> {
        // parameters
        input_size: input_size,
        output_size: output_size,
        window_function: window_function,

        // state
        row_index: 0,
        col_index: 0,
        start_mels_iter: start_mels_iter,
        end_mels_iter: end_mels_iter,
        window_size: first_window_size,
        window_iter: first_window_iter,
    }
}

/// one filter per row
pub fn enumerate_mel_scaling_matrix<T, T1, T2>(
    sample_rate: T1,
    window_size: T2,
    input_size: usize,
    output_size: usize,
) -> MelScalingMatrixEnumerator<T, apodize::TriangularWindowIter>
    where T: NumCast,
          T1: ToPrimitive,
          T2: ToPrimitive,
{
    let cast_sample_rate = <T as NumCast>::from(sample_rate)
        .expect("type `T` can't represent a certain value of type of `T1`.
                choose a `T1` that can represent fewer numbers
                or `T` that can represent more numbers.");
    let cast_window_size = <T as NumCast>::from(window_size)
        .expect("type `T` can't represent a certain value of type of `T2`.
                choose a `T2` that can represent fewer numbers
                or `T` that can represent more numbers.");
    enumerate_mel_scaling_matrix_base(
        hertz::rayleigh(cast_sample_rate, cast_window_size),
        hertz::nyquist(cast_sample_rate),
        input_size,
        output_size,
        mel_from_hertz,
        hertz_from_mel,
        apodize::triangular_iter
    )
}

// TODO test that identity matrix with
// output_size = input_size
// and conversion functions = identity

// TODO test in a small scale that scaling matrixes are correct
// then test it in emir directly and check correctness visually
