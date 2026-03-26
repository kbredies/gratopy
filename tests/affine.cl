// Simple test kernel for end-to-end custom kernel loading.

#ifdef pos_img_f
#undef pos_img_f
#undef pos_img_c
#endif

#define pos_img_f(x, y, z, Nx, Ny, Nz) (x + Nx * (y + Ny * z))
#define pos_img_c(x, y, z, Nx, Ny, Nz) (z + Nz * (y + Ny * x))

#ifdef real
#undef real
#endif
#define real \my_variable_type

__kernel void affine_\my_variable_type_\order1\order2(
    __global real *output,
    __global real *input
) {
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);

  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);

  size_t out_idx = pos_img_\order1(x, y, z, Nx, Ny, Nz);
  size_t in_idx = pos_img_\order2(x, y, z, Nx, Ny, Nz);

  output[out_idx] = (real)2 * input[in_idx] + (real)1;
}

__kernel void affine_ad_\my_variable_type_\order1\order2(
    __global real *output,
    __global real *input
) {
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);

  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);

  size_t out_idx = pos_img_\order1(x, y, z, Nx, Ny, Nz);
  size_t in_idx = pos_img_\order2(x, y, z, Nx, Ny, Nz);

  output[out_idx] = (real)3 * input[in_idx] - (real)1;
}
