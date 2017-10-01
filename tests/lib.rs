extern crate nalgebra;
use nalgebra::{DMatrix};

#[macro_use]
extern crate approx;

extern crate mel;
use mel::*;

#[test]
fn test_mel() {
    assert_relative_eq!(
        549.64, mel_from_hertz(440.), epsilon = 0.01);
    assert_relative_eq!(
        440., hertz_from_mel(549.64), epsilon = 0.01);

    let mel = 0.;
    assert_relative_eq!(mel, mel_from_hertz(hertz_from_mel(mel)));
    let mel = 100.;
    assert_relative_eq!(
        mel, mel_from_hertz(hertz_from_mel(mel)), epsilon = 0.0001);
    let mel = 3000.;
    assert_relative_eq!(
        mel, mel_from_hertz(hertz_from_mel(mel)), epsilon = 0.0001);

    let hertz = 0.;
    assert_relative_eq!(hertz, hertz_from_mel(mel_from_hertz(hertz)));
    let hertz = 1000.;
    assert_relative_eq!(hertz, hertz_from_mel(mel_from_hertz(hertz)), epsilon = 0.000001);
    let hertz = 44100. / 2.;
    assert_relative_eq!(hertz, hertz_from_mel(mel_from_hertz(hertz)));
}

#[test]
fn test_enumerate_mel_scaling_matrix() {
    let sample_rate = 44100;
    let window_size = 16;
    let power_spectrum_size = window_size / 2;
    let filter_count = 4;

    let mut actual = DMatrix::<f64>::zeros(filter_count, power_spectrum_size);

    for (row, col, coefficient) in enumerate_mel_scaling_matrix(
        sample_rate,
        window_size,
        power_spectrum_size,
        filter_count,
    ) {
        actual[(row, col)] = coefficient;
    }

    let expected = DMatrix::<f64>::from_row_slice(filter_count, power_spectrum_size, &[
        0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0.25, 0.25, 0., 0., 0., 0., 0.,
        0., 0., 0.11111111111111112, 0.3333333333333333, 0.11111111111111112, 0., 0., 0.,
        0., 0., 0., 0.039999999999999994, 0.12, 0.2, 0.12, 0.039999999999999994
    ]);

    for (expected, actual) in expected.iter().zip(actual.iter()) {
        assert_relative_eq!(expected, actual);
    }
}

// TODO test that identity matrix with
// output_size = input_size
// and conversion functions = identity

// TODO test in a small scale that scaling matrixes are correct
// then test it in emir directly and check correctness visually
