import numpy.typing as npt
import numba as nb


@nb.njit()
def linear_interpolation(
    x_current_start: float, 
    x_current_inverse_spacing: float,
    y_current_values: npt.NDArray,
    x_output_locations: npt.NDArray,
    y_output_values: npt.NDArray
) -> None:
    """Interpolate the input vector with nearest neighbors

    Args:

    Returns:
        None
    """    
    x_current_length: int = y_current_values.size - 1
    x_output_length: int = x_output_locations.size    
    
    # Loop over output location and compute the linear
    # contribution to the output pixel.
    for ndx in range(x_output_length):
        x_output = x_output_locations[ndx]
        
        input_ndx_float = (x_output - x_current_start) * x_current_inverse_spacing
        input_ndx = nb.int32(input_ndx_float)
        
        if (input_ndx >= 0) and (input_ndx < x_current_length):
            sub_pixel_offset = input_ndx_float - nb.float64(input_ndx)
            current_y = y_current_values[input_ndx]

            y_output_values[ndx] = (
                sub_pixel_offset * (y_current_values[input_ndx+1] - current_y)
                + current_y
            )
        
        else:
            y_output_values[ndx] = 0.0   