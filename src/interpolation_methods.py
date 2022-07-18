import numpy as np


__all__ = ['nearest_neighbors', 'linear_interpolation', 'cubic_keys', 'cubic_spline']


def nearest_neighbors(sampling_locations: np.ndarray):
    return np.where(np.abs(sampling_locations) < 0.5, 1 , 0)


def linear_interpolation(sampling_locations):
    return 1 - np.abs(sampling_locations)


def cubic_keys(sampling_locations):
    abs_sample_locations = np.abs(sampling_locations)
    
    def _under_one(val):
        return (
            1.5 * np.power(val, 3)
            - 2.5 * np.square(val) + 1
        )
    
    def _under_two(val):
        return (
            -0.5 * np.power(val, 3)
            + 2.5 * np.square(val) 
            - 4 * val + 2
        )
    
    mask1 = abs_sample_locations <= 1
    mask2 = (abs_sample_locations > 1) & (abs_sample_locations <= 2)
    
    return np.piecewise(abs_sample_locations, [mask1, mask2], [_under_one, _under_two])


def cubic_spline(sampling_locations):
    abs_sample_locations = np.abs(sampling_locations)
    
    def _under_one(val):
        return (
            0.5 * np.power(val, 3)
            - np.square(val) + 2 / 3
        )
    
    def _under_two(val):
        return (
            np.power(2 - val, 3) / 6
        )
    
    mask1 = abs_sample_locations <= 1
    mask2 = (abs_sample_locations > 1) & (abs_sample_locations <= 2)
    
    return np.piecewise(abs_sample_locations, [mask1, mask2], [_under_one, _under_two])