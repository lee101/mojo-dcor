"""Distance-correlation kernels and their C ABI."""

from std.math import pow, sqrt
from std.sys.info import simd_width_of

comptime Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime W = simd_width_of[DType.float64]()


def pointer(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


@always_inline
def powered_euclidean(
    a: Ptr, b: Ptr, dimensions: Int, exponent: Float64
) -> Float64:
    if dimensions == 1 and exponent == 1.0:
        return abs(a[unsafe_offset=0] - b[unsafe_offset=0])

    var accum = SIMD[DType.float64, W](0.0)
    var component = 0
    while component + W <= dimensions:
        var delta = a.unsafe_load[width=W](component) - b.unsafe_load[width=W](
            component
        )
        accum += delta * delta
        component += W

    var squared = accum.reduce_add()
    while component < dimensions:
        var delta = a[unsafe_offset=component] - b[unsafe_offset=component]
        squared += delta * delta
        component += 1

    if exponent == 1.0:
        return sqrt(squared)
    return pow(squared, exponent * 0.5)


@export("mdcor_stats")
def mdcor_stats(
    x_address: Int,
    y_address: Int,
    scratch_address: Int,
    result_address: Int,
    samples: Int,
    x_dimensions: Int,
    y_dimensions: Int,
    exponent: Float64,
    unbiased: Int,
) abi("C"):
    var x = pointer(x_address)
    var y = pointer(y_address)
    var scratch = pointer(scratch_address)
    var result = pointer(result_address)

    @__parameter
    def compute_row(row: Int):
        var x_sum = 0.0
        var y_sum = 0.0
        var xy_sum = 0.0
        var xx_sum = 0.0
        var yy_sum = 0.0
        for other in range(samples):
            var x_distance = powered_euclidean(
                x.unsafe_offset(row * x_dimensions),
                x.unsafe_offset(other * x_dimensions),
                x_dimensions,
                exponent,
            )
            var y_distance = powered_euclidean(
                y.unsafe_offset(row * y_dimensions),
                y.unsafe_offset(other * y_dimensions),
                y_dimensions,
                exponent,
            )
            x_sum += x_distance
            y_sum += y_distance
            xy_sum += x_distance * y_distance
            xx_sum += x_distance * x_distance
            yy_sum += y_distance * y_distance
        scratch[unsafe_offset=row] = x_sum
        scratch[unsafe_offset=samples + row] = y_sum
        scratch[unsafe_offset=2 * samples + row] = xy_sum
        scratch[unsafe_offset=3 * samples + row] = xx_sum
        scratch[unsafe_offset=4 * samples + row] = yy_sum

    for row in range(samples):
        compute_row(row)

    var x_total = 0.0
    var y_total = 0.0
    var row_product = 0.0
    var xy_total = 0.0
    var xx_total = 0.0
    var yy_total = 0.0
    for row in range(samples):
        var x_row = scratch[unsafe_offset=row]
        var y_row = scratch[unsafe_offset=samples + row]
        x_total += x_row
        y_total += y_row
        row_product += x_row * y_row
        xy_total += scratch[unsafe_offset=2 * samples + row]
        xx_total += scratch[unsafe_offset=3 * samples + row]
        yy_total += scratch[unsafe_offset=4 * samples + row]

    var covariance: Float64
    var variance_x: Float64
    var variance_y: Float64
    if unbiased != 0:
        var centered_denominator = Float64(samples - 2)
        var total_denominator = Float64((samples - 1) * (samples - 2))
        var normalization = Float64(samples * (samples - 3))
        covariance = (
            xy_total
            - 2.0 * row_product / centered_denominator
            + x_total * y_total / total_denominator
        ) / normalization

        var x_row_squared = 0.0
        var y_row_squared = 0.0
        for row in range(samples):
            x_row_squared += (
                scratch[unsafe_offset=row] * scratch[unsafe_offset=row]
            )
            y_row_squared += (
                scratch[unsafe_offset=samples + row]
                * scratch[unsafe_offset=samples + row]
            )
        variance_x = (
            xx_total
            - 2.0 * x_row_squared / centered_denominator
            + x_total * x_total / total_denominator
        ) / normalization
        variance_y = (
            yy_total
            - 2.0 * y_row_squared / centered_denominator
            + y_total * y_total / total_denominator
        ) / normalization
    else:
        var sample_count = Float64(samples)
        var normalization = sample_count * sample_count
        covariance = (
            xy_total
            - 2.0 * row_product / sample_count
            + x_total * y_total / normalization
        ) / normalization

        var x_row_squared = 0.0
        var y_row_squared = 0.0
        for row in range(samples):
            x_row_squared += (
                scratch[unsafe_offset=row] * scratch[unsafe_offset=row]
            )
            y_row_squared += (
                scratch[unsafe_offset=samples + row]
                * scratch[unsafe_offset=samples + row]
            )
        variance_x = (
            xx_total
            - 2.0 * x_row_squared / sample_count
            + x_total * x_total / normalization
        ) / normalization
        variance_y = (
            yy_total
            - 2.0 * y_row_squared / sample_count
            + y_total * y_total / normalization
        ) / normalization

    result[unsafe_offset=0] = covariance
    result[unsafe_offset=2] = variance_x
    result[unsafe_offset=3] = variance_y
    var denominator = sqrt(abs(variance_x * variance_y))
    result[unsafe_offset=1] = (
        covariance / denominator if denominator != 0.0 else 0.0
    )


@export("mdcor_pairwise_distances")
def mdcor_pairwise_distances(
    x_address: Int,
    destination_address: Int,
    samples: Int,
    dimensions: Int,
    exponent: Float64,
) abi("C"):
    var x = pointer(x_address)
    var destination = pointer(destination_address)

    @__parameter
    def compute_row(row: Int):
        for column in range(samples):
            destination[
                unsafe_offset=row * samples + column
            ] = powered_euclidean(
                x.unsafe_offset(row * dimensions),
                x.unsafe_offset(column * dimensions),
                dimensions,
                exponent,
            )

    for row in range(samples):
        compute_row(row)


@export("mdcor_center")
def mdcor_center(
    source_address: Int,
    matrix_address: Int,
    sums_address: Int,
    dimension: Int,
    unbiased: Int,
) abi("C"):
    var source = pointer(source_address)
    var matrix = pointer(matrix_address)
    var sums = pointer(sums_address)

    @__parameter
    def copy_and_sum_row(row: Int):
        var accum = SIMD[DType.float64, W](0.0)
        var column = 0
        var row_offset = row * dimension
        while column + W <= dimension:
            var values = source.unsafe_load[width=W](row_offset + column)
            matrix.unsafe_store(row_offset + column, values)
            accum += values
            column += W
        var row_sum = accum.reduce_add()
        while column < dimension:
            var value = source[unsafe_offset=row_offset + column]
            matrix[unsafe_offset=row_offset + column] = value
            row_sum += value
            column += 1
        sums[unsafe_offset=row] = row_sum

    for row in range(dimension):
        copy_and_sum_row(row)

    var total_accum = SIMD[DType.float64, W](0.0)
    var row = 0
    while row + W <= dimension:
        total_accum += sums.unsafe_load[width=W](row)
        row += W
    var total = total_accum.reduce_add()
    while row < dimension:
        total += sums[unsafe_offset=row]
        row += 1

    var axis_denominator = Float64(
        dimension - 2 if unbiased != 0 else dimension
    )
    var total_denominator = (
        Float64((dimension - 1) * (dimension - 2))
        if unbiased != 0
        else axis_denominator * axis_denominator
    )
    var reciprocal = 1.0 / axis_denominator
    var total_adjustment = total / total_denominator
    row = 0
    while row + W <= dimension:
        sums.unsafe_store(
            row, sums.unsafe_load[width=W](row) * reciprocal
        )
        row += W
    while row < dimension:
        sums[unsafe_offset=row] *= reciprocal
        row += 1

    @__parameter
    def center_row(row: Int):
        var row_offset = row * dimension
        var column = 0
        var row_adjustment = sums[unsafe_offset=row]
        while column + W <= dimension:
            matrix.unsafe_store(
                row_offset + column,
                matrix.unsafe_load[width=W](row_offset + column)
                - row_adjustment
                - sums.unsafe_load[width=W](column)
                + total_adjustment,
            )
            column += W
        while column < dimension:
            matrix[unsafe_offset=row_offset + column] = (
                matrix[unsafe_offset=row_offset + column]
                - row_adjustment
                - sums[unsafe_offset=column]
                + total_adjustment
            )
            column += 1
        if unbiased != 0:
            matrix[unsafe_offset=row_offset + row] = 0.0

    for row in range(dimension):
        center_row(row)


@always_inline
def dot_range(a: Ptr, b: Ptr, begin: Int, end: Int) -> Float64:
    var accum = SIMD[DType.float64, W](0.0)
    var index = begin
    while index + W <= end:
        accum += a.unsafe_load[width=W](index) * b.unsafe_load[width=W](index)
        index += W
    var total = accum.reduce_add()
    while index < end:
        total += a[unsafe_offset=index] * b[unsafe_offset=index]
        index += 1
    return total


@export("mdcor_mean_product")
def mdcor_mean_product(
    a_address: Int,
    b_address: Int,
    size: Int,
) abi("C") -> Float64:
    var a = pointer(a_address)
    var b = pointer(b_address)
    var total = dot_range(a, b, 0, size)
    return total / Float64(size)


@export("mdcor_u_product")
def mdcor_u_product(
    a_address: Int,
    b_address: Int,
    dimension: Int,
) abi("C") -> Float64:
    var a = pointer(a_address)
    var b = pointer(b_address)
    var size = dimension * dimension
    var total = dot_range(a, b, 0, size)
    return total / Float64(dimension * (dimension - 3))
