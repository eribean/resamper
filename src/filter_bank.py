import numpy as np

import interpolation_methods as im


__all__ = ['FilterBank']


class FilterBank:
    """Initialize filter bank for interpolating kernel."""
    METHODS = {
        "nearest": (im.nearest_neighbors, (1, 2)),
        "linear": (im.linear_interpolation, (64, 2)),
        "cubic": (im.cubic_keys, (64, 4)),
        "spline": (im.cubic_spline, (64, 4))
    }    
    
    def __init__(self, method: str | tuple, n_filters: int=None) -> None:
        """Constructor."""
        if isinstance(method, str):
            interpolation_method, method_args = self.METHODS[method]
        else:
            interpolation_method = method[0]
            method_args = (None, method[1])
        
        n_filters_used, filter_support = method_args
        
        if n_filters:
            n_filters_used = n_filters
            
        self.interpolation_method = interpolation_method
        self.n_support = filter_support
        self.n_filters = n_filters_used

    @staticmethod
    def create_sub_filters(mother_kernel: callable, n_sub_filters: int, n_support: int) -> np.ndarray:
        """Create a sampling array for efficient interpolation.

        Args:
            mother_kernel (callable): Callable function that takes in sampling locations
                and returns filter weights.
            n_sub_filter (int): Number of discrete filters
            n_support (int): Compactness of the mother kernel

        Returns:
            An array of length [n_sub_filters, n_support] with each row corresponding to the
                a specific filter
        """
        if n_support % 2:
            raise AssertionError('The support must be even.')

        filter_length = n_support * n_sub_filters
        filter_sample_spacing = 1 / n_sub_filters

        support_region = n_support / 2

        sampling_locations = np.ogrid[-support_region:support_region:filter_sample_spacing]

        filter_locations = (np.arange(n_sub_filters, 0 , -1)[:, None] 
                            + np.arange(n_support) * n_sub_filters) % filter_length

        filter_weights = mother_kernel(sampling_locations[filter_locations])

        return np.vstack((filter_weights, np.roll(filter_weights[0], 1)))        

    def __call__(self):
        """Create and return the filter bank."""
        filter_bank = self.create_sub_filters(self.interpolation_method, 
                                              self.n_filters, 
                                              self.n_support)

        filter_bank /= filter_bank.sum(1)[:, None]
        
        return filter_bank