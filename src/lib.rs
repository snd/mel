/*!
```
extern crate hertz;
extern crate mel;

fn main() {
    let sample_rate = 44100;
    let window_size = 4096;
    let power_spectrum_size = window_size / 2;
    let filter_count = 100;

    let mut calls: usize = 0;

    for (row, col, value) in mel::enumerate_mel_scaling_matrix(
        sample_rate,
        window_size,
        power_spectrum_size,
        filter_count,
    ) {
        calls += 1;
    }
    assert_eq!(calls, power_spectrum_size * filter_count);
}
```
*/

extern crate ndarray;
use ndarray::ArrayBase;

#[macro_use]
extern crate nalgebra;
use nalgebra::ApproxEq;

extern crate num;
use num::{Float, FromPrimitive};

extern crate hertz;
extern crate apodize;

extern crate itertools;
use itertools::linspace;

macro_rules! f64_from_usize {
    ($val:expr) => {
        <f64>::from_usize($val)
            .expect("type `f64` can't represent a specific value of type `usize` on this architecture.");
    }
}

macro_rules! usize_from_f64 {
    ($val:expr) => {
        <usize>::from_f64($val)
            .expect("type `usize` can't represent a specific value of type `f64` on this architecture.");
    }
}

#[inline]
pub fn hertz_from_mel(mel: f64) -> f64 {
    700. * ((10.).powf(mel / 2595.) - 1.)
}

#[inline]
pub fn mel_from_hertz(hertz: f64) -> f64 {
    2595. * (1. + hertz / 700.).log10()
}

#[test]
fn test_mel() {
    assert_approx_eq_eps!(
        549.64, mel_from_hertz(440.), 0.01);
    assert_approx_eq_eps!(
        440., hertz_from_mel(549.64), 0.01);

    let mel = 0.;
    assert_approx_eq!(mel, mel_from_hertz(hertz_from_mel(mel)));
    let mel = 100.;
    assert_approx_eq_eps!(
        mel, mel_from_hertz(hertz_from_mel(mel)), 0.0001);
    let mel = 3000.;
    assert_approx_eq_eps!(
        mel, mel_from_hertz(hertz_from_mel(mel)), 0.0001);

    let hertz = 0.;
    assert_approx_eq!(hertz, hertz_from_mel(mel_from_hertz(hertz)));
    let hertz = 1000.;
    assert_approx_eq!(hertz, hertz_from_mel(mel_from_hertz(hertz)));
    let hertz = 44100. / 2.;
    assert_approx_eq!(hertz, hertz_from_mel(mel_from_hertz(hertz)));
}

/// col index changes fastest
pub struct MelScalingMatrixEnumerator<WindowIter> {
    // parameters

    /// equivalent to col count
    input_size: usize,
    /// equivalent to row count
    output_size: usize,

    mel_from_hertz: fn(f64) -> f64,
    hertz_from_mel: fn(f64) -> f64,

    max_hertz: f64,

    window_function: fn(usize) -> WindowIter,

    // state

    row_index: usize,
    col_index: usize,

    start_mels_iter: itertools::Linspace<f64>,
    end_mels_iter: itertools::Linspace<f64>,

    // this gets set anew for every row
    window_start: usize,
    window_size: usize,
    window_iter: WindowIter
}

impl<WindowIter> MelScalingMatrixEnumerator<WindowIter>
    where WindowIter: Iterator<Item=f64>
{
    #[inline]
    pub fn is_done(&self) -> bool {
        self.is_after_last_row() && self.is_after_last_col()
    }

    #[inline]
    pub fn is_after_last_row(&self) -> bool {
        self.output_size <= self.row_index
    }

    #[inline]
    pub fn is_after_last_col(&self) -> bool {
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

impl<WindowIter> Iterator for MelScalingMatrixEnumerator<WindowIter>
    where WindowIter: Iterator<Item=f64>
{
    type Item = (usize, usize, f64);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize, f64)> {
        if self.is_done() {
            return None;
        }

        // start a new row/filter
        if self.is_after_last_col() {
            self.col_index = 0;
            self.row_index += 1;
            if self.is_after_last_row() {
                return None;
            }

            let start_mel = self.start_mels_iter.next()
                .expect("self.start_mels_iter.next()");
            let end_mel = self.end_mels_iter.next()
                .expect("self.end_mels_iter.next()");

            let hertz_from_mel = self.hertz_from_mel;
            let start_hertz = hertz_from_mel(start_mel);
            let end_hertz = hertz_from_mel(end_mel);

            // TODO maybe round or floor or ceil here
            self.window_start = usize_from_f64!(
                start_hertz /
                self.max_hertz *
                f64_from_usize!(self.output_size));

            let window_end = usize_from_f64!(
                end_hertz /
                self.max_hertz *
                f64_from_usize!(self.output_size));

            self.window_size = window_end - self.window_start;

            let window_function = self.window_function;
            self.window_iter = window_function(self.window_size);
        }

        let col = self.col_index;

        let value = if col < self.window_start {
            match self.window_iter.next() {
                Some(value) => {
                    value / f64_from_usize!(self.window_size)
                },
                None => 0.
            }
        } else {
            0.
        };

        self.col_index += 1;
        Some((self.row_index, col, value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl<WindowIter> ExactSizeIterator for MelScalingMatrixEnumerator<WindowIter>
    where WindowIter: Iterator<Item=f64>
{
    #[inline]
    fn len(&self) -> usize {
        self.input_size * self.output_size
    }
}

/// (or many vectors at once in form of a matrix)
/// to transform that vector into mel space.
/// efficiently
/// rows are mel
/// cols are freq
/// TODO get frequency index pair for nth mel
/// TODO enumerate scaling vector
/// TODO which dimension first
pub fn enumerate_mel_scaling_matrix_base<WindowIter>(
    // TODO more descriptive argument names
    min_hertz: f64,
    max_hertz: f64,
    // TODO shape
    input_size: usize,
    output_size: usize,
    mel_from_hertz: fn(f64) -> f64,
    hertz_from_mel: fn(f64) -> f64,
    window_function: fn(usize) -> WindowIter
) -> MelScalingMatrixEnumerator<WindowIter>
    where WindowIter: Iterator<Item=f64>
{
    assert!(min_hertz < max_hertz);
    assert!(output_size < input_size);
    assert!(0 < output_size);
    assert!(0 < input_size);

    let min_mel = mel_from_hertz(min_hertz);
    let max_mel = mel_from_hertz(max_hertz);

    // initially start_mels_iter == end_mels_iter
    let mut start_mels_iter = linspace(min_mel, max_mel, output_size + 1);
    let mut end_mels_iter = linspace(max_mel, max_mel, output_size + 1);
    end_mels_iter.next();

    let start_mel = start_mels_iter.next().unwrap();
    let end_mel = end_mels_iter.next().unwrap();

    let start_hertz = hertz_from_mel(start_mel);
    let end_hertz = hertz_from_mel(end_mel);

    // TODO maybe round or floor or ceil here
    let start_index = usize_from_f64!(
        start_hertz / max_hertz * f64_from_usize!(output_size));

    let end_index = usize_from_f64!(
        end_hertz / max_hertz * f64_from_usize!(output_size));

    let window_size = end_index - start_index;

    let window_iter = window_function(window_size);

    MelScalingMatrixEnumerator::<WindowIter> {
        // parameters
        input_size: input_size,
        output_size: output_size,
        window_function: window_function,
        hertz_from_mel: hertz_from_mel,
        mel_from_hertz: mel_from_hertz,
        max_hertz: max_hertz,

        // state
        row_index: 0,
        col_index: 0,

        start_mels_iter: start_mels_iter,
        end_mels_iter: end_mels_iter,

        window_start: start_index,
        window_size: window_size,
        window_iter: window_iter,
    }
}

/// one filter per row
pub fn enumerate_mel_scaling_matrix(
    sample_rate: usize,
    window_size: usize,
    input_size: usize,
    output_size: usize,
) -> MelScalingMatrixEnumerator<apodize::TriangularWindowIter>
{
    enumerate_mel_scaling_matrix_base(
        hertz::rayleigh(
            f64_from_usize!(sample_rate),
            f64_from_usize!(window_size)),
        hertz::nyquist(f64_from_usize!(sample_rate)),
        input_size,
        output_size,
        mel_from_hertz,
        hertz_from_mel,
        apodize::triangular_iter
    )
}

#[test]
fn test_enumerate_mel_scaling_matrix() {
    let sample_rate = 44100;
    let window_size = 100;
    let power_spectrum_size = window_size / 2;
    let filter_count = 10;

    let mut data: ArrayBase<Vec<f64>, (usize, usize)> =
        ArrayBase::zeros((filter_count, power_spectrum_size));

    for (row, col, value) in enumerate_mel_scaling_matrix(
        sample_rate,
        window_size,
        power_spectrum_size,
        filter_count,
    ) {
        data[(row, col)] = value;
    }
    println!("{:?}", data);
}

// TODO test that identity matrix with
// output_size = input_size
// and conversion functions = identity

// TODO test in a small scale that scaling matrixes are correct
// then test it in emir directly and check correctness visually
