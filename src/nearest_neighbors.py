import numpy.typing as npt
import numba as nb


@nb.njit()
def nearest_neighbors(
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
    x_current_length: int = y_current_values.size
    x_output_length: int = x_output_locations.size
    
    # Loop over output location and compute
    # the nearest input sample location and
    # take that value
    for ndx in range(x_output_length):
        x_output = x_output_locations[ndx]
        
        fractional_sample = nb.int32(
            (x_output - x_current_start) * x_current_inverse_spacing + 0.5
        )
        
        if (fractional_sample >= 0) and (fractional_sample < x_current_length):
            y_output_values[ndx] = y_current_values[fractional_sample]
        
        else:
            y_output_values[ndx] = 0.0
    