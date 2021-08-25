#include "eigen_ufunc.hh"
#include <cmath>

const char* pole_zero_doc = R"(
    Applies a Pole-zero correction using time constant tau
    Parameters
    ----------
    w_in : array-like
           waveform to apply pole zero correction to. Needs to be baseline subtracted
    
    t_tau : float
            Time constant of exponential decay to be deconvolved
    
    w_out : array-like
            Output array for pole zero corrected waveform 
    Processing Chain Example
    ------------------------
    "wf_pz": {
        "function": "pole_zero",
        "module": "pygama.dsp.processors",
        "args": ["wf_blsub", "db.pz.tau", "wf_pz"],
        "prereqs": ["wf_blsub"],
        "unit": "ADC",
        "defaults": { "db.pz.tau":"74*us" }
        },
)";

template <typename T, int A>
void pole_zero(const_wfblock_ref<T, A> w_in,
	       const_scalarblock_ref<T, A> t_tau,
	       wfblock_ref<T, A> w_out
	       ) {
  scalarblock<T, t_tau.RowsAtCompileTime> c(t_tau.rows());
  c = Eigen::exp(-t_tau.inverse());
  
  auto not_nan = w_in.isFinite().rowwise().all() && t_tau.isFinite();
  w_out.col(0) = not_nan.select(w_in.col(0), NAN);
  
  for(int i=1; i<w_out.cols(); i++)
    w_out.col(i) = w_out.col(i-1) + w_in.col(i) - w_in.col(i-1)*c;
}

add_ufunc_impl(pole_zero_f, (pole_zero<float, Aligned>), (pole_zero<float, Unaligned>) )
add_ufunc_impl(pole_zero_d, (pole_zero<double, Aligned>), (pole_zero<double, Unaligned>) )
create_ufunc(pole_zero_ufunc, "pole_zero", "(n),()->(n)", pole_zero_doc, pole_zero_f, pole_zero_d)
create_module(pole_zero, pole_zero_ufunc)
