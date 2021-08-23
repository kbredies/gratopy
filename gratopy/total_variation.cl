//
//    Copyright (C) 2021 Kristian Bredies (kristian.bredies@uni-graz.at)
//                       Richard Huber (richard.huber@uni-graz.at)
//
//    This file is part of gratopy (https://github.com/kbredies/gratopy).
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.

// Array indexing for C contiguous or Fortran contiguous arrays
#ifdef pos_img_f
#undef pos_img_f
#undef pos_sino_f
#undef pos_img_c
#undef pos_sino_c
#endif

#define pos_img_f(x, y, z, Nx, Ny, Nz) (x + Nx * (y + Ny * z))
#define pos_sino_f(s, a, z, Ns, Na, Nz) (s + Ns * (a + Na * z))
#define pos_img_c(x, y, z, Nx, Ny, Nz) (z + Nz * (y + Ny * x))
#define pos_sino_c(s, a, z, Ns, Na, Nz) (z + Nz * (a + Na * s))

#ifdef real
#undef real
#undef real2
#undef real3
#undef real8
#endif
#define real \my_variable_type
#define real2 \my_variable_type2
#define real3 \my_variable_type3
#define real8 \my_variable_type8

__kernel void empty_test_\my_variable_type_\order1() {
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z = get_global_id(2);

  size_t i = pos_img_\order1(x, y, z, Nx, Ny, Nz);
}

__kernel void update_v_\my_variable_type_\order1(__global real3 *v,
                                                 __global real *u,
                                                 const float sigma,
                                                 const float z_distance) {
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);
  size_t asdf = get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z = get_global_id(2);

  size_t i = pos_img_\order1(x, y, z, Nx, Ny, Nz);
  u += i;

  // gradient
  real3 val = -u[0];
  if (x < Nx - 1) {
    val.s0 += u[pos_img_\order1(1, 0, 0, Nx, Ny, Nz)];
  } else {
    val.s0 = (real)0.;
  }
  if (y < Ny - 1) {
    val.s1 += u[pos_img_\order1(0, 1, 0, Nx, Ny, Nz)];
  } else {
    val.s1 = (real)0.;
  }
  if (z < Nz - 1) {
    val.s2 += u[pos_img_\order1(0, 0, 1, Nx, Ny, Nz)];
  } else {
    val.s2 = (real)0.;
  }

  if (z_distance > 0) {
    val.s2 /= z_distance; // adjust to further Jump
  } else {
    val.s2 = (real)0.;
  }
  // step
  v[i] += sigma * (val);
}

__kernel void update_lambda_L2_\my_variable_type_\order1(
    __global real *lambda, __global real *Ku, __global real *f,
    const float sigma, const float sigmap1inv) {

  size_t Ns = get_global_size(0), Na = get_global_size(1),
         Nz = get_global_size(2);
  size_t s = get_global_id(0), a = get_global_id(1);
  size_t z = get_global_id(2);
  size_t i = pos_sino_\order1(s, a, z, Ns, Na, Nz);

  lambda[i] = (lambda[i] + sigma * (Ku[i] - f[i])) * sigmap1inv;
}

__kernel void update_u_\my_variable_type_\order1(
    __global real *u, __global real *u_, __global real3 *v,
    __global real *Kstarlambda, const float tau, const float norming,
    const float z_distance) {

  size_t Nx = get_global_size(0), Ny = get_global_size(1),
         Nz = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z = get_global_id(2);
  size_t i = pos_img_\order1(x, y, z, Nx, Ny, Nz);

  // divergence
  v += i;
  real3 val = v[0];

  if (x == Nx - 1)
    val.s0 = (real)0.;
  if (x > 0)
    val.s0 -= v[pos_img_\order1((-1), 0, 0, Nx, Ny, Nz)].s0;

  if (y == Ny - 1)
    val.s1 = (real)0.;
  if (y > 0)
    val.s1 -= v[pos_img_\order1(0, (-1), 0, Nx, Ny, Nz)].s1;

  if (z == Nz - 1)
    val.s2 = (real)0.;
  if (z > 0)
    val.s2 -= v[pos_img_\order1(0, 0, (-1), Nx, Ny, Nz)].s2;

  if (z_distance > 0) {
    val.s2 /= z_distance;
  } // adjust for further step
  else {
    val.s2 = 0;
  }

  // linear step
  u[i] = fmax((real)0., u_[i] + tau * (val.s0 + val.s1 + val.s2 -
                                       norming * Kstarlambda[i]));
}

__kernel void
    update_NormV_unchor_\my_variable_type_\order1(__global real3 *V,
                                                  __global real *normV) {

  size_t Nx = get_global_size(0), Ny = get_global_size(1),
         Nz = get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);

  size_t i = pos_img_\order1(x, y, z, Nx, Ny, Nz);

  // Compute norm
  real norm;
  real3 val = V[i];
  norm = hypot(hypot(val.s0, val.s1), val.s2);
  if (norm > (real)1.) {
    V[i] /= norm;
  }
}
