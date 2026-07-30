import numpy as np
import pytest

import dcor
import mojo_dcor as mdcor

rng = np.random.default_rng(20260730)


@pytest.mark.parametrize("dimensions", [1, 2, 7])
@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5])
def test_biased_statistics_match_upstream(dimensions, exponent):
    x = rng.normal(size=(37, dimensions))
    y = rng.normal(size=(37, dimensions + 1))
    got = mdcor.distance_stats_sqr(x, y, exponent=exponent)
    expected = dcor.distance_stats_sqr(
        x, y, exponent=exponent, method="naive"
    )
    assert tuple(got) == pytest.approx(tuple(expected), rel=3e-10, abs=3e-10)


def test_public_scalar_estimators_match_upstream():
    x = rng.normal(size=(45, 3))
    y = np.sin(x[:, :1]) + rng.normal(scale=0.1, size=(45, 1))
    names = [
        "distance_covariance_sqr",
        "distance_covariance",
        "distance_correlation_sqr",
        "distance_correlation",
    ]
    for name in names:
        got = getattr(mdcor, name)(x, y)
        expected = getattr(dcor, name)(x, y, method="naive")
        assert got == pytest.approx(expected, rel=2e-12, abs=2e-12)


def test_unsquared_stats_and_compatibility_types():
    x = rng.normal(size=(29, 2))
    y = rng.normal(size=(29, 3))
    got = mdcor.distance_stats(x, y)
    expected = dcor.distance_stats(x, y, method="naive")
    assert isinstance(got, mdcor.Stats)
    assert tuple(got) == pytest.approx(tuple(expected), rel=2e-12, abs=2e-12)
    assert mdcor.CompileMode.AUTO.name == "AUTO"
    assert mdcor.DistanceCovarianceMethod.NAIVE.name == "NAIVE"
    assert mdcor.EstimationStatistic.U_STATISTIC.name == "U_STATISTIC"


@pytest.mark.parametrize("dimensions", [1, 4])
def test_unbiased_statistics_match_upstream(dimensions):
    x = rng.normal(size=(31, dimensions))
    y = x[:, :1] ** 2 + rng.normal(scale=0.2, size=(31, 2))
    got = mdcor.u_distance_stats_sqr(x, y)
    expected = dcor.u_distance_stats_sqr(x, y, method="naive")
    assert tuple(got) == pytest.approx(tuple(expected), rel=3e-12, abs=3e-12)
    assert mdcor.u_distance_covariance_sqr(x, y) == pytest.approx(
        dcor.u_distance_covariance_sqr(x, y, method="naive"),
        rel=3e-12,
        abs=3e-12,
    )
    assert mdcor.u_distance_correlation_sqr(x, y) == pytest.approx(
        dcor.u_distance_correlation_sqr(x, y, method="naive"),
        rel=3e-12,
        abs=3e-12,
    )


def test_parallel_statistics_threshold_matches_upstream():
    x = rng.normal(size=(65, 3))
    y = rng.normal(size=(65, 5))
    got = mdcor.distance_stats_sqr(x, y)
    expected = dcor.distance_stats_sqr(x, y, method="naive")
    assert tuple(got) == pytest.approx(
        tuple(expected), rel=3e-12, abs=3e-12
    )


def test_published_upstream_example():
    a = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    b = np.array([[1.0], [0.0], [0.0], [1.0]])
    assert mdcor.distance_correlation(a, a) == pytest.approx(1.0)
    assert mdcor.distance_correlation(a, b) == pytest.approx(
        0.5266403878479265
    )
    assert mdcor.distance_correlation(a, b, exponent=0.5) == pytest.approx(
        0.6703214421008729
    )


def test_constant_input_has_zero_correlation():
    x = np.ones((20, 3))
    y = rng.normal(size=(20, 2))
    assert mdcor.distance_correlation_sqr(x, y) == 0
    assert mdcor.distance_correlation(x, y) == 0


def test_mismatched_samples_rejected():
    with pytest.raises(ValueError, match="same number"):
        mdcor.distance_correlation(np.ones((4, 1)), np.ones((5, 1)))


def test_lossy_input_conversions_are_rejected():
    with pytest.raises(TypeError, match="complex"):
        mdcor.distance_correlation(np.array([1 + 2j]), np.array([1 + 2j]))
    with pytest.raises(TypeError, match="precision loss"):
        mdcor.distance_correlation(
            np.arange(4, dtype=np.longdouble),
            np.arange(4, dtype=np.longdouble),
        )
    with pytest.raises(ValueError, match="represented exactly"):
        mdcor.distance_correlation(
            np.array([0, 2**53 + 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        )
    with pytest.raises(TypeError, match="real numeric"):
        mdcor.mean_product(np.array(["1"]), np.array(["1"]))
    with pytest.raises(TypeError, match="complex"):
        mdcor.double_centered(np.eye(2, dtype=np.complex128))


def test_noncontiguous_and_read_only_inputs_are_copied_safely():
    source = rng.normal(size=(20, 6))
    x = source[:, ::2]
    y = source[:, 1::2]
    x.flags.writeable = False
    assert mdcor.distance_correlation(x, y) == pytest.approx(
        dcor.distance_correlation(x, y, method="naive")
    )


def test_method_names_are_accepted_and_invalid_name_rejected():
    x = np.arange(8.0)
    baseline = mdcor.distance_correlation(x, x, method="naive")
    for method in ["auto", "avl", "mergesort"]:
        assert mdcor.distance_correlation(x, x, method=method) == baseline
    with pytest.raises(ValueError, match="Unknown"):
        mdcor.distance_correlation(x, x, method="bogus")


def test_outside_recommended_exponent_warns_like_upstream():
    x = np.arange(5.0)
    with pytest.warns(UserWarning, match="not guaranteed"):
        mdcor.distance_correlation(x, x, exponent=2)


def test_small_unbiased_sample_is_nan():
    for samples in [0, 1, 2, 3]:
        value = mdcor.u_distance_covariance_sqr(np.arange(float(samples)), np.arange(float(samples)))
        assert np.isnan(value)


def test_double_centered_matches_upstream_and_has_zero_sums():
    matrix = rng.normal(size=(13, 13))
    matrix = matrix + matrix.T
    got = mdcor.double_centered(matrix)
    expected = dcor.double_centered(matrix)
    assert got == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert got.sum(axis=0) == pytest.approx(0, abs=2e-14)
    assert got.sum(axis=1) == pytest.approx(0, abs=2e-14)


@pytest.mark.parametrize("function_name", ["double_centered", "u_centered"])
def test_centering_simd_tail_matches_upstream(function_name):
    matrix = rng.normal(size=(17, 17))
    matrix += matrix.T
    got = getattr(mdcor, function_name)(matrix)
    expected = getattr(dcor, function_name)(matrix)
    assert got == pytest.approx(expected, rel=2e-13, abs=2e-13)


def test_centering_out_returns_and_updates_same_array():
    matrix = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    destination = np.empty_like(matrix, dtype=np.float32)
    returned = mdcor.double_centered(matrix, out=destination)
    assert returned is destination
    assert destination == pytest.approx(dcor.double_centered(matrix))


def test_centering_can_use_float64_output_buffer_in_place():
    matrix = rng.normal(size=(17, 17))
    matrix += matrix.T
    expected = dcor.double_centered(matrix)

    destination = np.empty_like(matrix)
    returned = mdcor.double_centered(matrix, out=destination)
    assert returned is destination
    assert destination == pytest.approx(expected, rel=2e-13, abs=2e-13)

    returned = mdcor.double_centered(matrix, out=matrix)
    assert returned is matrix
    assert matrix == pytest.approx(expected, rel=2e-13, abs=2e-13)


def test_read_only_centering_output_is_rejected():
    destination = np.empty((4, 4))
    destination.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        mdcor.double_centered(np.ones((4, 4)), out=destination)


def test_u_centered_matches_upstream():
    matrix = rng.uniform(size=(15, 15))
    matrix = matrix + matrix.T
    got = mdcor.u_centered(matrix)
    expected = dcor.u_centered(matrix)
    assert got == pytest.approx(expected, rel=2e-13, abs=2e-13)
    assert np.diag(got) == pytest.approx(0)


def test_products_match_upstream():
    a = rng.normal(size=(19, 19))
    b = rng.normal(size=(19, 19))
    assert mdcor.mean_product(a, b) == pytest.approx(dcor.mean_product(a, b))
    assert mdcor.u_product(a, b) == pytest.approx(dcor.u_product(a, b))
    rectangular = rng.normal(size=(7, 11))
    assert mdcor.mean_product(rectangular, rectangular) == pytest.approx(
        dcor.mean_product(rectangular, rectangular)
    )


def test_product_simd_tail_matches_upstream():
    a = rng.normal(size=1003)
    b = rng.normal(size=1003)
    assert mdcor.mean_product(a, b) == pytest.approx(dcor.mean_product(a, b))


def test_projection_helpers_match_upstream():
    a = mdcor.u_centered(rng.normal(size=(9, 9)))
    b = mdcor.u_centered(rng.normal(size=(9, 9)))
    assert mdcor.u_projection(a)(b) == pytest.approx(dcor.u_projection(a)(b))
    assert mdcor.u_complementary_projection(a)(b) == pytest.approx(
        dcor.u_complementary_projection(a)(b)
    )


def test_partial_distance_statistics_match_upstream():
    x = rng.normal(size=(18, 2))
    y = x[:, :1] ** 2 + rng.normal(scale=0.3, size=(18, 1))
    z = rng.normal(size=(18, 3))
    assert mdcor.partial_distance_covariance(x, y, z) == pytest.approx(
        dcor.partial_distance_covariance(x, y, z), rel=2e-12, abs=2e-12
    )
    assert mdcor.partial_distance_correlation(x, y, z) == pytest.approx(
        dcor.partial_distance_correlation(x, y, z), rel=2e-12, abs=2e-12
    )


def test_affinely_invariant_statistics_match_upstream():
    x = rng.normal(size=(32, 4))
    y = rng.normal(size=(32, 2))
    assert mdcor.distance_correlation_af_inv_sqr(x, y) == pytest.approx(
        dcor.distance_correlation_af_inv_sqr(x, y, method="naive"),
        rel=2e-12,
        abs=2e-12,
    )
    assert mdcor.distance_correlation_af_inv(x, y) == pytest.approx(
        dcor.distance_correlation_af_inv(x, y, method="naive"),
        rel=2e-12,
        abs=2e-12,
    )
