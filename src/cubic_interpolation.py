import numpy.typing as npt
import numpy as np
import numba as nb


@nb.njit
def cubic_interpolation(
    x_current_start: float, 
    x_current_inverse_spacing: float,
    y_current_values: npt.NDArray,
    x_output_locations: npt.NDArray,
    y_output_values: npt.NDArray,
    cubic_filter: npt.NDArray
) -> None:
    """Interpolate the input vector with nearest neighbors

    Args:

    Returns:
        None
    """    
    x_current_length: int = y_current_values.size - 2
    x_output_length: int = x_output_locations.size    

    # Loop over output location and compute the
    # contribution to the output pixel.
    for ndx in range(x_output_length):
        x_output = x_output_locations[ndx]
        
        input_ndx_float = (x_output - x_current_start) * x_current_inverse_spacing
        input_ndx = nb.int32(input_ndx_float)
        
        if (input_ndx >= 1) and (input_ndx < x_current_length):
            sub_pixel_offset = input_ndx_float - nb.float64(input_ndx)

            sample_0 = cubic_filter[0, 0] + sub_pixel_offset * (
                cubic_filter[0, 1] + sub_pixel_offset * (
                    cubic_filter[0, 2] + sub_pixel_offset * cubic_filter[0, 3]
                )
            )

            sample_1 = cubic_filter[1, 0] + sub_pixel_offset * (
                cubic_filter[1, 1] + sub_pixel_offset * (
                    cubic_filter[1, 2] + sub_pixel_offset * cubic_filter[1, 3]
                )
            )

            sample_2 = cubic_filter[2, 0] + sub_pixel_offset * (
                cubic_filter[2, 1] + sub_pixel_offset * (
                    cubic_filter[2, 2] + sub_pixel_offset * cubic_filter[2, 3]
                )
            )

            sample_3 = cubic_filter[3, 0] + sub_pixel_offset * (
                cubic_filter[3, 1] + sub_pixel_offset * (
                    cubic_filter[3, 2] + sub_pixel_offset * cubic_filter[3, 3]
                )
            )

            y_output_values[ndx] = (
                sample_0 * y_current_values[input_ndx-1]
                + sample_1 * y_current_values[input_ndx]
                + sample_2 * y_current_values[input_ndx+1]
                + sample_3 * y_current_values[input_ndx+2]
            )
        
        else:
            y_output_values[ndx] = 0.0


class CubicKeys:
    """Filter coefficients for cubic keys interpolation."""
    needs_prefilter: bool = False

    coeffs: npt.NDArray[np.float64] = np.array([
        [0.0, -0.5,  1.0, -0.5],
        [1.0,  0.0  -2.5,  1.5],
        [0.0,  0.5,  2.0, -1.5],
        [0.0,  0.0, -0.5,  0.5]
    ])


class CubicSpline:
    """Filter coefficients for cubic spline interpolation."""
    needs_prefilter: bool = True

    prefilter: float = -0.2679491924311228

    coeffs: npt.NDArray[np.float64] = np.array([
        [1.0, -3.0,  3.0, -1.0],
        [4.0,  0.0, -6.0,  3.0],
        [1.0,  3.0,  3.0, -3.0],
        [0.0,  0.0,  0.0,  1.0]
    ])


class CubicOMOMS:
    """Filter coefficients for cubic spline interpolation."""
    needs_prefilter: bool = True

    prefilter: float = -0.34413115425505025

    coeffs: npt.NDArray[np.float64] = np.array([
        [1.00, -2.75,   2.625, -0.875],
        [3.25,  0.375, -5.25,   2.625],
        [1.00,  2.25,   2.625, -2.625],
        [0.00,  0.125,  0.00,   0.875]
    ]) 