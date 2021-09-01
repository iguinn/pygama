#include "eigen_ufunc.hh"

const char* mean_doc = R"(
    Calculates mean of waveform
    Parameters
    ----------
    w_in : array-like
           waveform take mean of
    
    a_out : float
            mean of w_in
)";

// get mean of waveform. Mostly just implemented as a test...
template <typename T, int A>
void mean(const_wfblock_ref<T, A> w_in, scalarblock_ref<T, A> a_out) {
  a_out = w_in.rowwise().mean();
}

add_ufunc_impl(mean_f, (mean<float, Aligned>), (mean<float, Unaligned>) )
add_ufunc_impl(mean_d, (mean<double, Aligned>), (mean<double, Unaligned>) )
create_ufunc(mean_ufunc, "mean", "(n)->()", mean_doc, mean_f, mean_d)
create_module(mean, mean_ufunc)
