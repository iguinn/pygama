"""
Contains a list of dsp processors used by the legend experiment, implemented
using numba's guvectorize to implement numpy's ufunc interface. In other words,
all of the functions are void functions whose outputs are given as parameters.
The ufunc interface provides additional information about the function
signatures that enables broadcasting the arrays and SIMD processing. Thanks to
the ufunc interface, they can also be called to return a numpy array, but if
 this is done, memory will be allocated for this array, slowing things down.
"""

# I think there's a way to do this recursively, but I'll figure it out later...
from ._processors.pole_zero import pole_zero, double_pole_zero
from ._processors.fixed_time_pickoff import fixed_time_pickoff
from ._processors.time_point_thresh import time_point_thresh
from ._processors.param_lookup import param_lookup
from ._processors.fftw import dft, inv_dft, psd
from ._processors.linear_slope_fit import linear_slope_fit
from ._processors.log_check import log_check
from ._processors.min_max import min_max
from ._processors.presum import presum
from ._processors.windower import windower
from ._processors.bl_subtract import bl_subtract
from ._processors.convolutions import cusp_filter, zac_filter, t0_filter
from ._processors.trap_filters import trap_filter, trap_norm, asym_trap_filter, trap_pickoff
from ._processors.moving_windows import moving_window_left, moving_window_right, moving_window_multi, avg_current
from ._processors.mean_stdev import mean_stdev
from ._processors.time_point_frac import time_point_frac
asymTrapFilter = asym_trap_filter

deprecated = {'time_point_frac': "Use time_point_thresh instead",
              'mean_stdev': "Use linear_slope_fit instead",
              'asymTrapFilter': "Use asym_trap_filter instead"
             }

depr_funs = { name:globals()[name] for name in deprecated.keys() }
for name in deprecated.keys():
    del globals()[name]


def __getattr__(name):
    if name in deprecated:
        print(name + " is deprecated: " + deprecated[name])
        return depr_funs[name]
    raise AttributeError

