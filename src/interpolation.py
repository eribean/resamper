import numpy as np
from numba.pycc import CC

cc = CC('resamper')

@cc.export('interp_1d', 'f8[:](f8[:], f8[:], f8[:], f8[:, :])')
def interpolate_1d(x_current, y_current, x_output, filter_bank):
    y_output = np.zeros_like(x_output)
    n_filters, n_support = filter_bank.shape
    n_filters -= 1
    start_ndx = 1 - n_support // 2
    x_start = x_current[0]
    x_spacing = 1 / (x_current[2] - x_current[1])
    
    for ndx, position in enumerate(x_output):
        ndx_position = (position - x_start) * x_spacing
        center_position = int(np.floor(ndx_position))
        filter_number = int(n_filters * (ndx_position - center_position) + 0.5)
        weights = filter_bank[filter_number]
        
        for ndx2 in range(n_support):
            input_ndx = center_position + ndx2 + start_ndx
            y_output[ndx] += y_current[input_ndx] * weights[ndx2]
     
    return y_output