import numpy as np
from numba.pycc import CC


cc = CC('resamper')


@cc.export('compiled_spline_prefilter', '(f8[:], float64)')
def spline_prefilter(input_array: np.ndarray, tau: float) -> None:
    """Apply a recursive prefilter for spline interpolation.

    A recursive high-pass filter necessary for spline interpolation. The
    filter is applied in-place, make a copy external to this function.

    Args:
        input_array (ndarray): one-dimensional numpy array
        tau (float): constant used in the recursive filter

    Returns:
        None

    Notes:
        A numba compiled version exists "compiled_spline_prefilter"
    """
    initialization_length = min(int(-12 / np.log10(abs(tau))), 
                                input_array.size)

    initial_condition = 0.0
    tau_power = 1.0

    # Initialization of recursive filter
    for ndx in range(initialization_length):
        initial_condition += input_array[ndx] * tau_power
        tau_power *= tau

    input_array[0] = initial_condition

    # Forward recursive filter
    for ndx in range(1, input_array.size-1):
        input_array[ndx] += tau * input_array[ndx-1]

    # Initialization of reverse filter
    input_array[-1] = (tau / (1 - tau * tau)
                       * (input_array[-1] +  tau * input_array[-2]))

    # Reverse recursive filter
    for ndx in range(input_array.size-2, -1, -1):
        input_array[ndx] = tau * (input_array[ndx+1] - input_array[ndx])


@cc.export('compiled_linear_prefilter', '(f8[:],)')
def linear_prefilter(input_array: np.ndarray) -> None:
    """Apply recursive filter for offset linear interpolation.
    
    """
    initialization_length = 18
    tau = 0.21

    initial_condition = 0.0
    tau_power = 1.0

    for ndx in range(initialization_length):
        initial_condition += input_array[ndx] * tau_power
        tau_power *= tau

    input_array[0] = initial_condition

    for ndx in range(1, input_array.size-1):
        input_array[ndx] += tau * (input_array[ndx] - input_array[ndx-1])
        

if __name__ == "__main__":
    cc.compile()