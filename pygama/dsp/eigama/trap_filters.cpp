#include <cmath>
#include "eigen_ufunc.hh"

const char* trap_doc = R"(
  Applies a symmetric trapezoidal filter (rise= fall) to the waveform
  Parameters:
  -----------
  w_in : array-like
         Input Waveform
  rise : float
         Sets the number of samples that will be averaged in the rise and fall sections
  flat : float
         Controls the delay between the rise and fall averaging sections, 
         typically around 3us for ICPC energy estimation, lower for detectors with shorter drift times
  w_out : array-like
          Output waveform after trap filter applied
  Processing Chain Example
  ------------------------
  "wf_trap": {
      "function": "trap_filter",
      "module": "pygama.dsp.processors",
      "args": ["wf_pz", "10*us", "3*us", "wf_trap"],
      "prereqs": ["wf_pz"],
      "unit": "ADC"
      },
)";

// recursive vectorized trap filter algorithm
template <typename T_wf, typename T_time, int A>
void trap_filter(const_wfblock_ref<T_wf, A> wf_in, T_time rise, T_time flat, wfblock_ref<T_wf, A> trap) {
  int rise_int = int(round(rise));
  int flat_int = int(round(flat));

  // NaN checking. TODO: add bounds checking...
  auto not_nan = wf_in.isFinite().rowwise().all();
  trap.col(0) = not_nan.select(wf_in.col(0), NAN);
  
  for(int i=1; i<rise_int; ++i) {
    trap.col(i) = trap.col(i-1) + wf_in.col(i); }
  for(int i=rise; i<rise_int+flat_int; ++i)
    trap.col(i) = trap.col(i-1) + wf_in.col(i) - wf_in.col(i-rise_int);
  for(int i=rise+flat; i<2*rise+flat; ++i)
    trap.col(i) = trap.col(i-1) + wf_in.col(i) - wf_in.col(i-rise_int) - wf_in.col(i-rise_int-flat_int);
  for(int i=2*rise_int+flat_int; i<trap.cols(); ++i)
    trap.col(i) = trap.col(i-1) + wf_in.col(i) - wf_in.col(i-rise_int) - wf_in.col(i-rise_int-flat_int) + wf_in.col(i-2*rise_int-flat_int);
}

add_ufunc_impl(trap_filter_fi, (trap_filter<float, int, Aligned>), (trap_filter<float, int, Unaligned>));
add_ufunc_impl(trap_filter_di, (trap_filter<double, int, Aligned>), (trap_filter<double, int, Unaligned>));
add_ufunc_impl(trap_filter_fd, (trap_filter<float, double, Aligned>), (trap_filter<float, double, Unaligned>));
add_ufunc_impl(trap_filter_dd, (trap_filter<double, double, Aligned>), (trap_filter<double, double, Unaligned>));
create_ufunc(trap_filter_ufunc, "trap_filter", "(n),(),()->(n)", trap_doc,
	     trap_filter_fi, trap_filter_di, trap_filter_fd, trap_filter_dd)


const char* trap_norm_doc = R"(
  Applies a symmetric trapezoidal filter (rise= fall) to the waveform
  Parameters:
  -----------
  w_in : array-like
         Input Waveform
  rise : float
         Sets the number of samples that will be averaged in the rise and fall sections
  flat : float
         Controls the delay between the rise and fall averaging sections, 
         typically around 3us for ICPC energy estimation, lower for detectors with shorter drift times
  w_out : array-like
          Output waveform after trap filter applied
  Processing Chain Example
  ------------------------
  "wf_trap": {
      "function": "trap_norm",
      "module": "pygama.dsp.processors",
      "args": ["wf_pz", "10*us", "3*us", "wf_trap"],
      "prereqs": ["wf_pz"],
      "unit": "ADC"
      },
)";

// recursive vectorized trap filter algorithm
template <typename T_wf, typename T_time, int A>
void trap_norm(const_wfblock_ref<T_wf, A> wf_in, T_time rise, T_time flat, wfblock_ref<T_wf, A> trap) {
  trap.col(0) = wf_in.col(0);
  int rise_int = int(round(rise));
  int flat_int = int(round(flat));
  
  // NaN checking. TODO: add bounds checking...
  auto not_nan = wf_in.isFinite().rowwise().all();
  trap.col(0) = not_nan.select(wf_in.col(0), NAN);
  
  for(int i=1; i<rise_int; ++i)
    trap.col(i) = (trap.col(i-1) + wf_in.col(i))/rise;
  for(int i=rise; i<rise_int+flat_int; ++i)
    trap.col(i) = (trap.col(i-1) + wf_in.col(i) - wf_in.col(i-rise_int))/rise;
  for(int i=rise+flat; i<2*rise+flat; ++i)
    trap.col(i) = (trap.col(i-1) + wf_in.col(i) - wf_in.col(i-rise_int) - wf_in.col(i-rise_int-flat_int))/rise;
  for(int i=2*rise_int+flat_int; i<trap.cols(); ++i)
    trap.col(i) = (trap.col(i-1) + wf_in.col(i) - wf_in.col(i-rise_int) - wf_in.col(i-rise_int-flat_int) + wf_in.col(i-2*rise_int-flat_int))/rise;
}

add_ufunc_impl(trap_norm_fi, (trap_norm<float, int, Aligned>), (trap_norm<float, int, Unaligned>));
add_ufunc_impl(trap_norm_di, (trap_norm<double, int, Aligned>), (trap_norm<double, int, Unaligned>));
add_ufunc_impl(trap_norm_fd, (trap_norm<float, double, Aligned>), (trap_norm<float, double, Unaligned>));
add_ufunc_impl(trap_norm_dd, (trap_norm<double, double, Aligned>), (trap_norm<double, double, Unaligned>));
create_ufunc(trap_norm_ufunc, "trap_norm", "(n),(),()->(n)", trap_norm_doc,
	     trap_norm_fi, trap_norm_di, trap_norm_fd, trap_norm_dd)



create_module(trap_filters, trap_filter_ufunc, trap_norm_ufunc)
