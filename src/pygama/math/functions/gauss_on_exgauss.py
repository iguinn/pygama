r"""
Provide a convenience function for a Gaussian on top of an extended Gaussian. The correct usage is :func:`gauss_on_exgauss.get_pdf(x, mu, sigma, htail, tau)`

Parameters
----------
mu
    The location and scale of the first Gaussian
htail
    The fraction of the Gaussian tail vs. the Gaussian peak
tau
    The characteristic scale of the extended Gaussian tail

Returns
-------
gauss_on_exgauss
    A subclass of :class:`sum_dists` and :class:`rv_continuous`, has methods of :func:`pdf`, :func:`cdf`, etc.

Notes
-----
The extended Gaussian distribution shares the mu, sigma with the Gaussian
"""
from pygama.math.functions.sum_dists import sum_dists


from pygama.math.functions.gauss import gaussian
from pygama.math.functions.exgauss import exgauss

(mu, sigma, frac, tau) = range(4) # the sum_dist array we will pass will be mu, sigma, frac/htail, tau
par_array = [(exgauss, [tau, mu, sigma]), (gaussian, [mu, sigma])] 

gauss_on_exgauss = sum_dists(par_array, [frac], "fracs", parameter_names = ["mu", "sigma", "htail", "tau"])