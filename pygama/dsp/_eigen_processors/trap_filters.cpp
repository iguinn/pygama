#include <iostream>
#include "eigen_ufunc.hh"

// recursive vectorized trap filter algorithm
template <typename T_wf, typename T_time, int Align>
void trap(const_wfblock_ref<T_wf, Align> wf_in, T_time rise, T_time flat, wfblock_ref<T_wf, Align> trap) {
  trap.col(0) = wf_in.col(0);
  int rise_int = int(round(rise));
  int flat_int = int(round(flat));
  
  for(int i=1; i<rise_int; ++i) {
    trap.col(i) = trap.col(i-1) + wf_in.col(i); }
  for(int i=rise; i<rise_int+flat_int; ++i)
    trap.col(i) = trap.col(i-1) + wf_in.col(i) - wf_in.col(i-rise_int);
  for(int i=rise+flat; i<2*rise+flat; ++i)
    trap.col(i) = trap.col(i-1) + wf_in.col(i) - wf_in.col(i-rise_int) - wf_in.col(i-rise_int-flat_int);
  for(int i=2*rise_int+flat_int; i<trap.cols(); ++i)
    trap.col(i) = trap.col(i-1) + wf_in.col(i) - wf_in.col(i-rise_int) - wf_in.col(i-rise_int-flat_int) + wf_in.col(i-2*rise_int-flat_int);
}

add_ufunc_impl(trap_fi, (trap<float, int, align>), (trap<float, int, Eigen::Unaligned>));
add_ufunc_impl(trap_di, (trap<double, int, align>), (trap<double, int, Eigen::Unaligned>));
add_ufunc_impl(trap_fd, (trap<float, double, align>), (trap<float, double, Eigen::Unaligned>));
add_ufunc_impl(trap_dd, (trap<double, double, align>), (trap<double, double, Eigen::Unaligned>));
create_ufunc(trap_ufunc, "trap", "trap(w_in[], t_rise, t_flat, w_out[])",
	     trap_fi, trap_di, trap_fd, trap_dd)
create_module(trap_filters, trap_ufunc)
