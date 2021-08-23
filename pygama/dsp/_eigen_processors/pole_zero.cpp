#include <iostream>
#include "eigen_ufunc.hh"
#include <cmath>

// recursive vectorized pole-zero filter algorithm
template <typename T, int Align>
void pole_zero(const_wfblock_ref<T, Align> w_in,
	       const_scalarblock_ref<T, Align> t_tau,
	       wfblock_ref<T, Align> w_out
	       ) {
  scalarblock<T, t_tau.RowsAtCompileTime> c(t_tau.rows());
  c = Eigen::exp(-t_tau.inverse());

  w_out.col(0) = w_in.col(0);
  for(int i=1; i<w_out.cols(); i++)
    w_out.col(i) = w_out.col(i-1) + w_in.col(i) - w_in.col(i-1)*c;
}

add_ufunc_impl(pole_zero_f, (pole_zero<float, align>), (pole_zero<float, Eigen::Unaligned>) )
add_ufunc_impl(pole_zero_d, (pole_zero<double, align>), (pole_zero<double, Eigen::Unaligned>) )
create_ufunc(pole_zero_ufunc, "pole_zero", "pole_zero(w_in[], t_tau, w_out[])", pole_zero_f, pole_zero_d)
create_module(pole_zero, pole_zero_ufunc)
