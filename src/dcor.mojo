"""Distance-correlation kernels and their C ABI."""

from std.algorithm import sync_parallelize
from std.math import pow, sqrt
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()


def pointer(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


@always_inline
def powered_euclidean(a: Ptr, b: Ptr, dimensions: Int, exponent: Float64) -> Float64:
    if dimensions == 1 and exponent == 1.0:
        return abs(a[0] - b[0])

    var accum = SIMD[DType.float64, W](0.0)
    var component = 0
    while component + W <= dimensions:
        var delta = (
            a.load[width=W](component) - b.load[width=W](component)
        )
        accum += delta * delta
        component += W

    var squared = accum.reduce_add()
    while component < dimensions:
        var delta = a[component] - b[component]
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

    @parameter
    def compute_row(row: Int):
        var x_sum = 0.0
        var y_sum = 0.0
        var xy_sum = 0.0
        var xx_sum = 0.0
        var yy_sum = 0.0
        for other in range(samples):
            var x_distance = powered_euclidean(
                x + row * x_dimensions,
                x + other * x_dimensions,
                x_dimensions,
                exponent,
            )
            var y_distance = powered_euclidean(
                y + row * y_dimensions,
                y + other * y_dimensions,
                y_dimensions,
                exponent,
            )
            x_sum += x_distance
            y_sum += y_distance
            xy_sum += x_distance * y_distance
            xx_sum += x_distance * x_distance
            yy_sum += y_distance * y_distance
        scratch[row] = x_sum
        scratch[samples + row] = y_sum
        scratch[2 * samples + row] = xy_sum
        scratch[3 * samples + row] = xx_sum
        scratch[4 * samples + row] = yy_sum

    if samples >= 64:
        sync_parallelize[compute_row](samples)
    else:
        for row in range(samples):
            compute_row(row)

    var x_total = 0.0
    var y_total = 0.0
    var row_product = 0.0
    var xy_total = 0.0
    var xx_total = 0.0
    var yy_total = 0.0
    for row in range(samples):
        var x_row = scratch[row]
        var y_row = scratch[samples + row]
        x_total += x_row
        y_total += y_row
        row_product += x_row * y_row
        xy_total += scratch[2 * samples + row]
        xx_total += scratch[3 * samples + row]
        yy_total += scratch[4 * samples + row]

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
            x_row_squared += scratch[row] * scratch[row]
            y_row_squared += (
                scratch[samples + row] * scratch[samples + row]
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
            x_row_squared += scratch[row] * scratch[row]
            y_row_squared += (
                scratch[samples + row] * scratch[samples + row]
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

    result[0] = covariance
    result[2] = variance_x
    result[3] = variance_y
    var denominator = sqrt(abs(variance_x * variance_y))
    result[1] = covariance / denominator if denominator != 0.0 else 0.0


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

    @parameter
    def compute_row(row: Int):
        for column in range(samples):
            destination[row * samples + column] = powered_euclidean(
                x + row * dimensions,
                x + column * dimensions,
                dimensions,
                exponent,
            )

    if samples >= 64:
        sync_parallelize[compute_row](samples)
    else:
        for row in range(samples):
            compute_row(row)


@export("mdcor_center")
def mdcor_center(
    matrix_address: Int,
    sums_address: Int,
    dimension: Int,
    unbiased: Int,
) abi("C"):
    var matrix = pointer(matrix_address)
    var sums = pointer(sums_address)
    for row in range(dimension):
        var accum = SIMD[DType.float64, W](0.0)
        var column = 0
        var row_offset = row * dimension
        while column + W <= dimension:
            accum += matrix.load[width=W](row_offset + column)
            column += W
        var row_sum = accum.reduce_add()
        while column < dimension:
            row_sum += matrix[row_offset + column]
            column += 1
        sums[row] = row_sum

    var total_accum = SIMD[DType.float64, W](0.0)
    var row = 0
    while row + W <= dimension:
        total_accum += sums.load[width=W](row)
        row += W
    var total = total_accum.reduce_add()
    while row < dimension:
        total += sums[row]
        row += 1

    @parameter
    def center_row(row: Int):
        var row_offset = row * dimension
        var column = 0
        if unbiased != 0:
            var axis_denominator = Float64(dimension - 2)
            var total_denominator = Float64((dimension - 1) * (dimension - 2))
            var row_adjustment = sums[row] / axis_denominator
            var total_adjustment = total / total_denominator
            while column + W <= dimension:
                matrix.store(
                    row_offset + column,
                    matrix.load[width=W](row_offset + column)
                    - row_adjustment
                    - sums.load[width=W](column) / axis_denominator
                    + total_adjustment,
                )
                column += W
            while column < dimension:
                matrix[row_offset + column] = (
                    matrix[row_offset + column]
                    - row_adjustment
                    - sums[column] / axis_denominator
                    + total_adjustment
                )
                column += 1
            matrix[row_offset + row] = 0.0
        else:
            var axis_denominator = Float64(dimension)
            var total_denominator = axis_denominator * axis_denominator
            var row_adjustment = sums[row] / axis_denominator
            var total_adjustment = total / total_denominator
            while column + W <= dimension:
                matrix.store(
                    row_offset + column,
                    matrix.load[width=W](row_offset + column)
                    - row_adjustment
                    - sums.load[width=W](column) / axis_denominator
                    + total_adjustment,
                )
                column += W
            while column < dimension:
                matrix[row_offset + column] = (
                    matrix[row_offset + column]
                    - row_adjustment
                    - sums[column] / axis_denominator
                    + total_adjustment
                )
                column += 1

    for row in range(dimension):
        center_row(row)


@always_inline
def dot_range(a: Ptr, b: Ptr, begin: Int, end: Int) -> Float64:
    var accum = SIMD[DType.float64, W](0.0)
    var index = begin
    while index + W <= end:
        accum += a.load[width=W](index) * b.load[width=W](index)
        index += W
    var total = accum.reduce_add()
    while index < end:
        total += a[index] * b[index]
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
