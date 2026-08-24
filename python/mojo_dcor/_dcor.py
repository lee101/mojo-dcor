"""Distance covariance and correlation with Mojo pairwise kernels."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from enum import Enum, auto
from typing import Callable, Iterator
import warnings

import numpy as np

from ._lib import address, f64, lib


class CompileMode(Enum):
    AUTO = auto()
    NO_COMPILE = auto()
    COMPILE_CPU = auto()
    COMPILE_PARALLEL = auto()


class DistanceCovarianceMethod(Enum):
    AUTO = auto()
    NAIVE = auto()
    AVL = auto()
    MERGESORT = auto()


class EstimationStatistic(Enum):
    U_STATISTIC = auto()
    V_STATISTIC = auto()


@dataclass(frozen=True)
class Stats:
    covariance_xy: np.float64
    correlation_xy: np.float64
    variance_x: np.float64
    variance_y: np.float64

    def __iter__(self) -> Iterator[np.float64]:
        return iter(astuple(self))


def _samples(value: object, name: str) -> np.ndarray:
    array = f64(value)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    return array


def _validate_options(
    exponent: float,
    method: DistanceCovarianceMethod | str,
    compile_mode: CompileMode,
) -> None:
    if not 0 < exponent < 2:
        warnings.warn(
            "Distance covariance is not guaranteed to characterize "
            "independence if the exponent value is not in the range (0, 2). "
            f"The exponent passed is {exponent}.",
            UserWarning,
            stacklevel=3,
        )
    if isinstance(method, str):
        if method.lower() not in {"auto", "naive", "avl", "mergesort"}:
            raise ValueError(f"Unknown distance covariance method {method!r}")
    elif not isinstance(method, DistanceCovarianceMethod):
        raise TypeError("method must be a DistanceCovarianceMethod or its name")
    if not isinstance(compile_mode, CompileMode):
        raise TypeError("compile_mode must be a CompileMode")


def _stats_sqr(
    x: object,
    y: object,
    *,
    exponent: float,
    method: DistanceCovarianceMethod | str,
    compile_mode: CompileMode,
    unbiased: bool,
) -> Stats:
    _validate_options(exponent, method, compile_mode)
    x_array = _samples(x, "x")
    y_array = _samples(y, "y")
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            "x and y must have the same number of observations "
            f"({x_array.shape[0]} != {y_array.shape[0]})"
        )
    samples = x_array.shape[0]
    if samples == 0 or (unbiased and samples < 4):
        nan = np.float64(np.nan)
        return Stats(nan, nan, nan, nan)

    scratch = np.empty(5 * samples, dtype=np.float64)
    result = np.empty(4, dtype=np.float64)
    lib().mdcor_stats(
        address(x_array),
        address(y_array),
        address(scratch, writable=True),
        address(result, writable=True),
        samples,
        x_array.shape[1],
        y_array.shape[1],
        float(exponent),
        int(unbiased),
    )
    return Stats(*(np.float64(value) for value in result))


def distance_stats_sqr(
    x: object,
    y: object,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats:
    return _stats_sqr(
        x,
        y,
        exponent=exponent,
        method=method,
        compile_mode=compile_mode,
        unbiased=False,
    )


def distance_stats(
    x: object,
    y: object,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats:
    stats = distance_stats_sqr(
        x, y, exponent=exponent, method=method, compile_mode=compile_mode
    )
    return Stats(
        np.sqrt(stats.covariance_xy),
        np.sqrt(stats.correlation_xy),
        np.sqrt(stats.variance_x),
        np.sqrt(stats.variance_y),
    )


def distance_covariance_sqr(
    x: object,
    y: object,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> np.float64:
    return distance_stats_sqr(
        x, y, exponent=exponent, method=method, compile_mode=compile_mode
    ).covariance_xy


def distance_covariance(
    x: object,
    y: object,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> np.float64:
    return np.sqrt(
        distance_covariance_sqr(
            x, y, exponent=exponent, method=method, compile_mode=compile_mode
        )
    )


def distance_correlation_sqr(
    x: object,
    y: object,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> np.float64:
    return distance_stats_sqr(
        x, y, exponent=exponent, method=method, compile_mode=compile_mode
    ).correlation_xy


def distance_correlation(
    x: object,
    y: object,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> np.float64:
    return np.sqrt(
        distance_correlation_sqr(
            x, y, exponent=exponent, method=method, compile_mode=compile_mode
        )
    )


def u_distance_stats_sqr(
    x: object,
    y: object,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> Stats:
    return _stats_sqr(
        x,
        y,
        exponent=exponent,
        method=method,
        compile_mode=compile_mode,
        unbiased=True,
    )


def u_distance_covariance_sqr(
    x: object,
    y: object,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> np.float64:
    return u_distance_stats_sqr(
        x, y, exponent=exponent, method=method, compile_mode=compile_mode
    ).covariance_xy


def u_distance_correlation_sqr(
    x: object,
    y: object,
    *,
    exponent: float = 1,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> np.float64:
    return u_distance_stats_sqr(
        x, y, exponent=exponent, method=method, compile_mode=compile_mode
    ).correlation_xy


def _distance_matrix(value: object, exponent: float = 1) -> np.ndarray:
    array = _samples(value, "x")
    samples, dimensions = array.shape
    destination = np.empty((samples, samples), dtype=np.float64)
    if samples:
        lib().mdcor_pairwise_distances(
            address(array),
            address(destination, writable=True),
            samples,
            dimensions,
            float(exponent),
        )
    return destination


def _centered(
    a: object, *, out: np.ndarray | None, unbiased: bool
) -> np.ndarray:
    source = np.asarray(a)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("a must be a square matrix")
    source = f64(source)
    direct_out = False
    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError("out must be a NumPy array")
        if out.shape != source.shape:
            raise ValueError("out must have the same shape as a")
        if not np.issubdtype(out.dtype, np.floating):
            raise TypeError("out must have a floating-point dtype")
        if not out.flags.writeable:
            raise ValueError("out must be writable")
        same_buffer = (
            source.dtype == np.float64
            and source.flags.c_contiguous
            and out.dtype == np.float64
            and out.flags.c_contiguous
            and address(source) == address(out)
        )
        direct_out = (
            out.dtype == np.float64
            and out.flags.c_contiguous
            and (same_buffer or not np.shares_memory(source, out))
        )

    if direct_out:
        work = out
    else:
        work = np.empty(source.shape, dtype=np.float64, order="C")

    dimension = work.shape[0]
    if dimension:
        sums = np.empty(dimension, dtype=np.float64)
        lib().mdcor_center(
            address(source),
            address(work, writable=True),
            address(sums, writable=True),
            dimension,
            int(unbiased),
        )
    if out is None or direct_out:
        return work
    np.copyto(out, work, casting="unsafe")
    return out


def double_centered(a: object, *, out: np.ndarray | None = None) -> np.ndarray:
    return _centered(a, out=out, unbiased=False)


def u_centered(a: object, *, out: np.ndarray | None = None) -> np.ndarray:
    return _centered(a, out=out, unbiased=True)


def _same_shape_arrays(a: object, b: object) -> tuple[np.ndarray, np.ndarray]:
    a_array = f64(a)
    b_array = f64(b)
    if a_array.shape != b_array.shape:
        raise ValueError("a and b must have the same shape")
    return a_array, b_array


def mean_product(a: object, b: object) -> np.float64:
    a_array, b_array = _same_shape_arrays(a, b)
    if a_array.size == 0:
        return np.float64(np.nan)
    return np.float64(
        lib().mdcor_mean_product(
            address(a_array), address(b_array), a_array.size
        )
    )


def u_product(a: object, b: object) -> np.float64:
    a_array, b_array = _same_shape_arrays(a, b)
    if (
        a_array.ndim != 2
        or a_array.shape[0] != a_array.shape[1]
    ):
        raise ValueError("a and b must be square matrices")
    dimension = a_array.shape[0]
    if dimension == 0:
        return np.float64(np.nan)
    return np.float64(
        lib().mdcor_u_product(
            address(a_array), address(b_array), dimension
        )
    )


def u_projection(a: object) -> Callable[[object], np.ndarray]:
    basis = f64(a)
    denominator = u_product(basis, basis)

    def projection(value: object) -> np.ndarray:
        array = f64(value)
        if denominator == 0:
            return np.zeros_like(basis)
        return u_product(array, basis) / denominator * basis

    return projection


def u_complementary_projection(a: object) -> Callable[[object], np.ndarray]:
    projection = u_projection(a)

    def complementary(value: object) -> np.ndarray:
        array = f64(value)
        return array - projection(array)

    return complementary


def partial_distance_covariance(
    x: object, y: object, z: object
) -> np.float64:
    a = u_centered(_distance_matrix(x))
    b = u_centered(_distance_matrix(y))
    c = u_centered(_distance_matrix(z))
    projection = u_complementary_projection(c)
    return u_product(projection(a), projection(b))


def partial_distance_correlation(
    x: object, y: object, z: object
) -> np.float64:
    a = u_centered(_distance_matrix(x))
    b = u_centered(_distance_matrix(y))
    c = u_centered(_distance_matrix(z))
    aa, bb, cc = u_product(a, a), u_product(b, b), u_product(c, c)
    ab, ac, bc = u_product(a, b), u_product(a, c), u_product(b, c)

    def correlation(product: np.float64, left: np.float64, right: np.float64):
        denominator_sqr = left * right
        if denominator_sqr == 0:
            return denominator_sqr
        return np.clip(product / np.sqrt(denominator_sqr), -1, 1)

    r_xy = correlation(ab, aa, bb)
    r_xz = correlation(ac, aa, cc)
    r_yz = correlation(bc, bb, cc)
    denominator = np.sqrt(1 - r_xz**2) * np.sqrt(1 - r_yz**2)
    return np.float64(
        (r_xy - r_xz * r_yz) / denominator if denominator != 0 else denominator
    )


def _af_inv_scaled(value: object) -> np.ndarray:
    array = _samples(value, "x")
    centered = array - np.mean(array, axis=0, keepdims=True)
    covariance = centered.T @ centered / (array.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues[eigenvalues <= 0] = np.inf
    inverse_sqrt = (eigenvectors * (1 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return (array @ inverse_sqrt) / (array.shape[0] - 1)


def distance_correlation_af_inv_sqr(
    x: object,
    y: object,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> np.float64:
    correlation = distance_correlation_sqr(
        _af_inv_scaled(x),
        _af_inv_scaled(y),
        method=method,
        compile_mode=compile_mode,
    )
    return np.float64(0 if np.isnan(correlation) else correlation)


def distance_correlation_af_inv(
    x: object,
    y: object,
    method: DistanceCovarianceMethod | str = DistanceCovarianceMethod.AUTO,
    compile_mode: CompileMode = CompileMode.AUTO,
) -> np.float64:
    return np.sqrt(
        distance_correlation_af_inv_sqr(
            x, y, method=method, compile_mode=compile_mode
        )
    )
