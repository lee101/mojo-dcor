# mojo-dcor

Distance correlation and covariance for Python, with the quadratic pairwise
work implemented in [Mojo](https://www.modular.com/mojo).

The package is a compatible replacement for the covered part of
[`dcor`](https://github.com/vnmabus/dcor):

```python
import numpy as np
import mojo_dcor as dcor

rng = np.random.default_rng(7)
x = rng.uniform(-1, 1, size=1000)
y = x**2 + rng.normal(0, 0.01, size=1000)

print(dcor.distance_correlation(x, y))
print(dcor.u_distance_correlation_sqr(x, y))
```

Input observations may be one-dimensional arrays or sample matrices; noncontiguous
arrays are copied safely and `x` and `y` may have different feature counts. Real
numeric inputs are normalized to float64. Complex values, extended-precision
floats, and integers outside float64's exact range are rejected instead of being
silently narrowed.

## Coverage

The following upstream names and signatures are covered:

| area | API |
| --- | --- |
| biased estimators | `distance_covariance`, `distance_covariance_sqr`, `distance_correlation`, `distance_correlation_sqr`, `distance_stats`, `distance_stats_sqr` |
| unbiased estimators | `u_distance_covariance_sqr`, `u_distance_correlation_sqr`, `u_distance_stats_sqr` |
| matrix operations | `double_centered`, `u_centered`, `mean_product`, `u_product` |
| projections and partial measures | `u_projection`, `u_complementary_projection`, `partial_distance_covariance`, `partial_distance_correlation` |
| affine-invariant measures | `distance_correlation_af_inv`, `distance_correlation_af_inv_sqr` |
| compatibility types | `Stats`, `CompileMode`, `DistanceCovarianceMethod`, `EstimationStatistic` |

`exponent` is supported, including the same warning outside the recommended
open interval `(0, 2)`. Algorithm names `auto`, `naive`, `avl`, and `mergesort`
are accepted so covered calls do not need rewriting. They currently select the
same parallel, memory-efficient naive kernel. In particular, the upstream
O(n log n) AVL and mergesort algorithms for one-dimensional samples are not
implemented yet; upstream `dcor` with its default `auto` method may be faster
for large one-dimensional inputs.

Energy distance, homogeneity and independence tests, permutation tests,
`rowwise`, and Python Array API backends are outside this port. NumPy arrays are
the supported Python storage type. Affine-invariant scaling uses
`numpy.linalg.eigh`; its distance-statistics work still runs in Mojo.

## Install

The repository carries a pinned Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-dcor.so`. The Python wrapper also rebuilds
the library when the Mojo source is newer. Set `MOJO_DCOR_LIB` to load a
prebuilt shared library from another location.

## Performance

Measured with `pixi run bench` on this machine, using the pinned Python and
`dcor` environments and the same float64 arrays for both implementations. The
upstream distance-statistics
cases explicitly use `method="naive"` because that is the algorithm implemented
here.

| case | mojo-dcor | dcor 0.7 | result |
| --- | ---: | ---: | ---: |
| `distance_correlation` (1k x 5/3) | 8.91 ms | 187.13 ms | 21.00x faster |
| `u_distance_covariance_sqr` (1.5k x 2) | 32.02 ms | 1053.83 ms | 32.91x faster |
| `double_centered` (1.5k x 1.5k) | 5.86 ms | 9.22 ms | 1.57x faster |
| `mean_product` (2k x 2k) | 7.37 ms | 17.25 ms | 2.34x faster |

Centering uses target-width SIMD for row reductions, scratch normalization, and
matrix updates, with scalar remainder loops. The input copy is fused into the
row-reduction pass, so a contiguous float64 NumPy input stays zero-copy across
the FFI boundary and the kernel avoids a separate full-matrix copy. A measured
parallel experiment was slower at both 1,500 squared and 3,000 squared because
this streaming kernel saturates memory bandwidth and pays two scheduling
barriers, so the production path stays serial. The estimator kernels avoid
materializing two distance matrices.

No GPU path is included: centering performs only a few arithmetic operations
per 16-24 bytes moved, far below the roughly 2-flop-per-byte threshold where a
device transfer can pay off. The high-compute distance kernels are already far
more than 5x ahead of upstream on the benchmark and were deliberately left
unchanged.

These are measured values, not projections; rerun the benchmark for the numbers
on another machine.

## How it works

Python normalizes each input once to C-contiguous float64 and allocates five
length-`n` scratch vectors. A single ctypes call passes their addresses and
shapes as integers across the C ABI. Mojo reconstructs
`UnsafePointer[Float64, AnyOrigin[mut=True]]` values, computes each pairwise row
in parallel, and reduces the row sums into the centered covariance and variance
terms.

The estimator uses O(n) auxiliary memory rather than storing O(n²) distance
matrices. Operations whose API returns a centered matrix necessarily allocate
that row-major n-by-n result. Memory remains Python-owned on both paths: Mojo
does not allocate, retain pointers, or manage object lifetimes.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

Tests assert numerical and behavioral parity against the real upstream
`dcor` 0.7 package, including multivariate and non-unit-exponent inputs,
biased and unbiased estimates, matrix helpers, partial measures, and published
documentation examples.

## License

MIT
