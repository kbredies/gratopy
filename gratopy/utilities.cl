// Array indexing for C contiguous or Fortran contiguous arrays
#define pos_img_f(x,y,z,Nx,Ny,Nz) (x+y*Nx+z*Nx*Ny)
#define pos_sino_f(s,a,z,Ns,Na,Nz) (s+a*Ns+Ns*Na*z)
#define pos_img_c(x,y,z,Nx,Ny,Nz) (z+y*Nz+x*Nz*Ny)
#define pos_sino_c(s,a,z,Ns,Na,Nz) (z+a*Nz+s*Nz*Na)




__kernel void multiply_\my_variable_type_\order1(__global \my_variable_type* sinogram,
                  __constant \my_variable_type *angle_weights,
                  __global \my_variable_type* sinogram_new)
{
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = get_global_size(2);

  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = get_global_id(2);


  sinogram_new[pos_sino_\order1(s,a,z,Ns,Na,Nz)] =
          sinogram[pos_sino_\order1(s,a,z,Ns,Na,Nz)] * angle_weights[a];
}

__kernel void divide_\my_variable_type_\order1(__global \my_variable_type* sinogram,
                  __constant \my_variable_type *angle_weights,
                  __global \my_variable_type* sinogram_new)
{
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = get_global_size(2);

  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = get_global_id(2);


  sinogram_new[pos_sino_\order1(s,a,z,Ns,Na,Nz)] =
          sinogram[pos_sino_\order1(s,a,z,Ns,Na,Nz)] / angle_weights[a];
}


__kernel void equ_mul_add_\my_variable_type_\order1( __global \my_variable_type *rhs,
                          \my_variable_type a, __global \my_variable_type* x)
{
size_t i = get_global_id(0);
rhs[i]+=a*x[i];
}

__kernel void mul_add_add_\my_variable_type_\order1(__global \my_variable_type *rhs,
                          \my_variable_type a,
                          __global \my_variable_type* x,
                          __global \my_variable_type *y)
{
size_t i = get_global_id(0);
rhs[i]=a*x[i]+y[i];
}
