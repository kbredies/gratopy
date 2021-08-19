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
#endif
#define real \my_variable_type

__kernel void multiply_\my_variable_type_\order1(__global real *sinogram,
                                                 __constant real *angle_weights,
                                                 __global real *sinogram_new) {
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = get_global_size(2);

  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = get_global_id(2);

  sinogram_new[pos_sino_\order1(s, a, z, Ns, Na, Nz)] =
      sinogram[pos_sino_\order1(s, a, z, Ns, Na, Nz)] * angle_weights[a];
}

__kernel void divide_\my_variable_type_\order1(__global real *sinogram,
                                               __constant real *angle_weights,
                                               __global real *sinogram_new) {
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = get_global_size(2);

  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = get_global_id(2);

  size_t i = pos_sino_\order1(s, a, z, Ns, Na, Nz);

  sinogram_new[i] = sinogram[i] / angle_weights[a];
}

__kernel void equ_mul_add_\my_variable_type_\order1(__global real *rhs, real a,
                                                    __global real *x) {
  size_t i = get_global_id(0);
  rhs[i] += a * x[i];
}

__kernel void mul_add_add_\my_variable_type_\order1(__global real *rhs, real a,
                                                    __global real *x,
                                                    __global real *y) {
  size_t i = get_global_id(0);
  rhs[i] = a * x[i] + y[i];
}
