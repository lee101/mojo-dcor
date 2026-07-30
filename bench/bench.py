"""Benchmarks against dcor 0.7 on identical arrays."""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import dcor  # noqa: E402
import mojo_dcor as mdcor  # noqa: E402


def time_best(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    rng = np.random.default_rng(47)
    cases = []

    x = rng.normal(size=(1_000, 5))
    y = x[:, :3] ** 2 + rng.normal(size=(1_000, 3))
    cases.append(
        (
            "distance_correlation (1k x 5/3)",
            lambda x=x, y=y: mdcor.distance_correlation(x, y),
            lambda x=x, y=y: dcor.distance_correlation(
                x, y, method="naive"
            ),
        )
    )

    x = rng.normal(size=(1_500, 2))
    y = rng.normal(size=(1_500, 2))
    cases.append(
        (
            "u_distance_covariance_sqr (1.5k x 2)",
            lambda x=x, y=y: mdcor.u_distance_covariance_sqr(x, y),
            lambda x=x, y=y: dcor.u_distance_covariance_sqr(
                x, y, method="naive"
            ),
        )
    )

    matrix = rng.normal(size=(1_500, 1_500))
    matrix += matrix.T
    cases.append(
        (
            "double_centered (1.5k x 1.5k)",
            lambda: mdcor.double_centered(matrix),
            lambda: dcor.double_centered(matrix),
        )
    )

    a = rng.normal(size=(2_000, 2_000))
    b = rng.normal(size=(2_000, 2_000))
    cases.append(
        (
            "mean_product (2k x 2k)",
            lambda: mdcor.mean_product(a, b),
            lambda: dcor.mean_product(a, b),
        )
    )

    mdcor.distance_correlation(np.arange(8.0), np.arange(8.0))
    print(
        f"{'case':<41}{'mojo-dcor':>13}{'dcor 0.7':>13}{'speedup':>11}"
    )
    print("-" * 78)
    for name, ours, upstream in cases:
        ours_time = time_best(ours)
        upstream_time = time_best(upstream)
        ratio = upstream_time / ours_time
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"{name:<41}{ours_time * 1e3:>11.2f}ms"
            f"{upstream_time * 1e3:>11.2f}ms"
            f"{ratio:>9.2f}x {label}"
        )

if __name__ == "__main__":
    main()
