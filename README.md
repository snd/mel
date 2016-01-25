# mel

*status: early work in progress. api still in flux.*

<!--
[![Build Status](https://travis-ci.org/snd/mel.svg?branch=master)](https://travis-ci.org/snd/mel/branches)
-->
<!--
[![](https://meritbadge.herokuapp.com/mel)](https://crates.io/crates/mel)
-->

**convert scalars and vectors from hertz to [mel scale](https://en.wikipedia.org/wiki/Mel_scale).
written in [rust](https://www.rust-lang.org/)**

useful for transforming a power spectrum vector into a smaller
and more meaningful representation.

- [x] `mel_from_hertz`
- [x] `hertz_from_mel`
- [ ] generate triangular windows that can map a frequency vector into a mel vector
- [ ] fill matrix that can be multiplied with a frequency vector to transform it into mel scale

<!--
to use add `mel = "*"`
to the `[dependencies]` section of your `Cargo.toml` and call `extern crate mel;` in your code.
-->

<!--
## [read the documentation for an example and more !](https://snd.github.io/mel/mel/index.html)
-->

### [contributing](contributing.md)

### licensed under either of [apache-2.0](LICENSE-APACHE) ([tl;dr](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0))) or [MIT](LICENSE-MIT) ([tl;dr](https://tldrlegal.com/license/mit-license)) at your option
