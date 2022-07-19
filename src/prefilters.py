import numpy as np
from numba.pycc import CC


cc = CC('resamper')


@cc.export('compiled_prefilter', '(f8[:], float64)')
def spline_prefilter(input_array: np.ndarray, tau: float) -> None:
    """Apply the cubic spline prefilter.
    
    
    """
    initialization_length = 20

    temp = 0.0
    tau_power = 1.0

    for ndx in range(initialization_length):
        temp += input_array[ndx] * tau_power
        tau_power *= tau

    input_array[0] = temp

    for ndx in range(1, input_array.size-1):
        input_array[ndx] += tau * input_array[ndx-1]

    input_array[-1] = (tau / (1 - tau * tau)
                       * (input_array[-1] +  tau * input_array[-2]))

    for ndx in range(input_array.size-2, -1, -1):
        input_array[ndx] = tau * (input_array[ndx+1] - input_array[ndx])


@cc.export('compiled_linear_prefilter', '(f8[:],)')
def offset_linear_prefilter(input_array: np.ndarray) -> None:
    """Apply recursive filter for offset linear interpolation.
    
    """
    initialization_length = 18
    tau = 0.21

    temp = 0.0
    tau_power = 1.0

    for ndx in range(initialization_length):
        temp += input_array[ndx] * tau_power
        tau_power *= tau

    input_array[0] = temp

    for ndx in range(1, input_array.size-1):
        input_array[ndx] += tau * (input_array[ndx] - input_array[ndx-1])
        

if __name__ == "__main__":
    cc.compile()