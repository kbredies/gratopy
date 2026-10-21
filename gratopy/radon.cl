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
#undef real4
#undef real8
#endif
#define real \my_variable_type
#define real2 \my_variable_type2
#define real3 \my_variable_type3
#define real4 \my_variable_type4
#define real8 \my_variable_type8

// Radon Transform
// Computes the forward projection in parallel beam geometry for a given image.
// the \my_variable_type_\order1\order2 suffix sets the kernel to the suitable
// precision, contiguity of the sinogram, contiguity of image.
// Input:
//			sino: Pointer to array representing sinogram (to be
//                            computed) with detector-dimension times
//                            angle-dimension times z dimension
// 			img:  Pointer to array representing image to be transformed
//			      of dimensions Nx times Ny times Nz
//                            (img_shape=Nx times Ny)
//                      ofs:  Buffer containing geometric informations concerning the
//			      projection-directions (angular information)
//                            Entries are cos, sin, offset and 1/max(|cos|,|sino|)
//			Geometryinformation: Contains various geometric
//                                           information
//                                           [delta_x, delta_xi, Nx, Ny, Ns, Na]
// Output:
//			values inside sino are altered to represent the computed
//                      Radon transform
__kernel void radon_\my_variable_type_\order1\order2(
    __global real *sino, __global real *img, __constant real8 *ofs,
    __constant real *Geometryinformation) {
  // Extract dimensions
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = get_global_size(2);
  const int Nx = Geometryinformation[2];
  const int Ny = Geometryinformation[3];

  // Extract current position
  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = get_global_id(2);

  // Extract scales
  const real delta_x = Geometryinformation[0];
  const real delta_xi = Geometryinformation[1];

  // hack (since otherwise s is unsigned which leads to overflow problems)
  int ss = s;

  // Extract angular information
  // o = (cos,sin,offset,1/max(|cos|,|sin|))
  real4 o = ofs[a].s0123;
  real reverse_mask = ofs[a].s5;

  // Dummy variable for switching from horizontal to vertical lines
  int Nxx = Nx;
  int Nyy = Ny;
  int horizontal = 1;

  // When line is horizontal rather than vertical, switch x and y dimensions
  if (reverse_mask != (real)0.) {
    horizontal = 0;
    o.xy = (real2)(o.y, o.x);

    Nxx = Ny;
    Nyy = Nx;
  }

  // accumulation variable
  real acc = (real)0.;

  // shift image to correct z-dimension (as this will remain fixed),
  // particularly relevant for "F" contiguity of image
  __global real *img0 = img + pos_img_\order2(0, 0, z, Nx, Ny, Nz);

  // stride representing one index_step in x dimension (dependent on
  // horizontal/vertical)
  size_t stride_x = horizontal == 1 ? pos_img_\order2(1, 0, 0, Nx, Ny, Nz)
                                    : pos_img_\order2(0, 1, 0, Nx, Ny, Nz);

  // for through the entire y dimension
  for (int y = 0; y < Nyy; y++) {
    int x_low, x_high;

    // project (0,y) onto detector
    real d = y * o.y + o.z - ss;

    // compute bounds
    x_low = (int)((-1 - d) * o.w);
    x_high = (int)((1 - d) * o.w);

    // case the direction is decreasing switch high and low
    if (o.w < (real)0.) {
      int trade = x_low;
      x_low = x_high;
      x_high = trade;
    }

    // make sure x inside image dimensions
    x_low = max(x_low, 0);
    x_high = min(x_high, Nxx - 1);

    // shift position of image depending on horizontal/vertical
    if (horizontal == 1)
      img = img0 + pos_img_\order2(x_low, y, 0, Nx, Ny, Nz);
    if (horizontal == 0)
      img = img0 + pos_img_\order2(y, x_low, 0, Nx, Ny, Nz);

    // integration in x dimension for fixed y
    for (int x = x_low; x <= x_high; x++) {
      // anterpolation weight via normal distance
      real weight = (real)1. - fabs(x * o.x + d);
      if (weight > (real)0.) {
        acc += weight * img[0];
      }
      // update image to next position
      img += stride_x;
    }
  }
  // assign value to sinogram
  sino[pos_sino_\order1(s, a, z, Ns, Na, Nz)] =
      acc * delta_x * delta_x / delta_xi;
}

// Radon backprojection
// Computes the backprojection projection in parallel beam geometry for a given
// image. the \my_variable_type_\order1\order2 suffix sets the kernel to the
// suitable precision, contiguity of the sinogram, contiguity of image.
// Input:
// 			img:  Pointer to array representing image (to be computed)
//			      of dimensions Nx times Ny times Nz
//                            (img_shape=Nx times Ny)
//			sino: Pointer to array representing sinogram (to be
//                            transformed) with detector-dimension times
//                            angle-dimension times z dimension
//                      ofs:  Buffer containing geometric informations concerning the
//			      projection-directions (angular information)
//                            Entries are cos, sin, offset and 1/max(|cos|,|sino|)
//			Geometryinformation: Contains various geometric
//                                           information
//                                           [delta_x, delta_xi, Nx, Ny, Ns, Na]
// Output:
//			values inside img are altered to represent the computed
//                      Radon backprojection
__kernel void radon_ad_\my_variable_type_\order1\order2(
    __global real *img, __global real *sino, __constant real8 *ofs,
    __constant real *Geometryinformation) {
  // Extract dimensions
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);
  const int Ns = Geometryinformation[4];
  const int Na = Geometryinformation[5];

  // Extract current position
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);

  // Accumulation variable
  real acc = (real)0.;
  real2 c = (real2)(x, y);

  // shift sinogram to correct z-dimension (as this will remain fixed),
  // particularly relevant for "F" contiguity of image
  sino += pos_sino_\order2(0, 0, z, Ns, Na, Nz);

  // Integrate with respect to angular dimension
  for (int a = 0; a < Na; a++) {
    // Extract angular dimensions
    real8 o = ofs[a];
    real Delta_phi = o.s4; // angle_width asociated to the angle

    // compute detector position associated to (x,y) and phi=a
    real s = dot(c, o.s01) + o.s2;

    // compute adjacent detector positions
    int sm = floor(s);
    int sp = sm + 1;

    // compute corresponding weights
    real weightp = 1 - (sp - s);
    real weightm = 1 - (s - sm);

    // set weight to zero in case adjacent detector position is outside
    // the detector range
    if (sm < 0 || sm >= Ns) {
      weightm = (real)0.;
      sm = 0;
    }
    if (sp < 0 || sp >= Ns) {
      weightp = (real)0.;
      sp = 0;
    }

    // accumulate weigthed sum (Delta_Phi weight due to angular resolution)
    acc += Delta_phi * (weightm * sino[pos_sino_\order2(sm, a, 0, Ns, Na, Nz)] +
                        weightp * sino[pos_sino_\order2(sp, a, 0, Ns, Na, Nz)]);
  }

  // Assign value to img
  img[pos_img_\order1(x, y, z, Nx, Ny, Nz)] = acc;
}

// Single Line of Radon Transform: Computes the Fanbeam transform of an image
// with delta peak in (x,y) Computes the forward projection in parallel beam
// geometry for a given image. the \my_variable_type_\order1\order2 suffix sets
// the kernel to the suitable precision, contiguity of the sinogram, contiguity
// of image.
// Input:
//			sino: Pointer to array representing sinogram (to be
//                            computed) with detector-dimension times
//                            angle-dimension times z dimension
// 			x,y:  Position of the delta peak to be transformed
//                      ofs:  Buffer containing geometric informations concerning the
//			      projection-directions (angular information)
//                            Entries are cos, sin, offset and 1/max(|cos|,|sino|)
//			Geometryinformation: Contains various geometric
//                                           information
//                                           [delta_x, delta_xi, Nx, Ny, Ns, Na]
// Output:
//			values inside sino are altered to represent the computed
//                      Radon transform obtained by transforming an image with
//                      Dirac-delta at (x,y)
__kernel void single_line_radon_\my_variable_type_\order1\order2(
    __global real *sino, int x, int y, __constant real8 *ofs,
    __constant real *Geometryinformation) {
  // Geometric dimensions
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = 1;
  const int Nx = Geometryinformation[2];
  const int Ny = Geometryinformation[3];

  // Extract current position
  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = 0;

  // Discretization parameters
  const real delta_x = Geometryinformation[0];
  const real delta_xi = Geometryinformation[1];

  // hack (since otherwise s is unsigned which leads to overflow problems)
  int ss = s;

  // Extract angular information
  // o = (cos,sin,offset,1/cos)
  real4 o = ofs[a].s0123;
  real reverse_mask = ofs[a].s5;
  // Dummy variable in case of vertical/horizontal switch
  int Nxx = Nx;
  int Nyy = Ny;

  // In case rays are vertical rather than horizontal, swap x and y dimensions
  int horizontal = 1;
  if (reverse_mask != (real)0.) {
    horizontal = 0;
    o.xy = (real2)(o.y, o.x);
    Nxx = Ny;
    Nyy = Nx;

    real trade = x;
    x = y;
    y = trade;
  }

  // accumulation variable
  real acc = (real)0.;

  int x_low, x_high;

  // project (0,y) onto detector
  real d = y * o.y + o.z;

  // compute bounds
  x_low = (int)((ss - 1 - d) * o.w);
  x_high = (int)((ss + 1 - d) * o.w);

  // In case the detector moves download, switch low and upper bound
  if (o.w < (real)0.) {
    int trade = x_low;
    x_low = x_high;
    x_high = trade;
  }

  // make sure x inside image dimensions
  x_low = max(x_low, 0);
  x_high = min(x_high, Nxx - 1);

  // integration in x dimension for fixed y
  if ((x_low <= x) && (x <= x_high)) {
    // anterpolation weight via normal distance
    real weight = (real)1. - fabs(x * o.x + d - ss);
    if (weight > (real)0.) {
      acc = weight;
    }
  }

  // assign value to sinogram
  sino[pos_sino_\order1(s, a, z, Ns, Na, Nz)] =
      acc * delta_x * delta_x / delta_xi;
}

// Helper function for ray-driven transforms
real ray_weightfkt_\my_variable_type_\order1\order2(
    real t, real kappa, real s_under, real s_upper, real difference) {
  real rhs = (real)0.;
  real epsilon = (real)0.0001;
  if (fabs(difference) < epsilon * s_under &&
      fabs(t) < s_upper * ((real)1. + epsilon)) {
    rhs = (real)1.;
    if (s_upper - fabs(t) < epsilon * s_upper) {
      rhs = (real)0.5;
    }
  } else if (fabs(t) < s_under) {
    rhs = (s_upper - s_under) / difference * kappa;
  } else if (fabs(t) < s_upper) {
    rhs = (s_upper - fabs(t)) / difference * kappa;
  }
  return rhs;
}

// ray-driven Radon Transform
// Computes the forward projection in parallel beam geometry for a given image.
// the \my_variable_type_\order1\order2 suffix sets the kernel to the suitable
// precision, contiguity of the sinogram, contiguity of image.
// Input:
//			sino: Pointer to array representing sinogram (to be
//                            computed) with detector-dimension times
//                            angle-dimension times z dimension
// 			img:  Pointer to array representing image to be transformed
//			      of dimensions Nx times Ny times Nz
//                            (img_shape=Nx times Ny)
//                      ofs:  Buffer containing geometric informations concerning the
//			      projection-directions (angular information)
//                            Entries are cos, sin, offset and 1/max(|cos|,|sino|)
//			Geometryinformation: Contains various geometric
//                                           information
//                                           [delta_x, delta_xi, Nx, Ny, Ns, Na]
// Output:
//			values inside sino are altered to represent the computed
//                      Radon transform
__kernel void radon_ray_\my_variable_type_\order1\order2(
    __global real *sino, __global real *img, __constant real8 *ofs,
    __constant real *Geometryinformation) {
  // Extract dimensions
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = get_global_size(2);
  const int Nx = Geometryinformation[2];
  const int Ny = Geometryinformation[3];

  // Extract current position
  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = get_global_id(2);

  // Extract scales
  const real delta_x = Geometryinformation[0];
  const real delta_xi = Geometryinformation[1];

  // hack (since otherwise s is unsigned which leads to overflow problems)
  int ss = s;

  // Extract angular information
  // o = (cos,sin,offset,1/max(|cos|,|sin|))
  real4 o = ofs[a].s0123;
  real reverse_mask = ofs[a].s5;
  real s_under = fabs(fabs(o.x) - fabs(o.y)) * (real)0.5;
  real s_upper = (fabs(o.x) + fabs(o.y)) * (real)0.5;
  real difference = s_upper - s_under;

  // Dummy variable for switching from horizontal to vertical lines
  int Nxx = Nx;
  int Nyy = Ny;
  int horizontal = 1;

  // When line is horizontal rather than vertical, switch x and y dimensions
  if (reverse_mask != (real)0.) {
    horizontal = 0;
    o.xy = (real2)(o.y, o.x);

    Nxx = Ny;
    Nyy = Nx;
  }

  // accumulation variable
  real acc = (real)0.;

  // shift image to correct z-dimension (as this will remain fixed),
  // particularly relevant for "F" contiguity of image
  __global real *img0 = img + pos_img_\order2(0, 0, z, Nx, Ny, Nz);

  // stride representing one index_step in x dimension (dependent on
  // horizontal/vertical)
  size_t stride_x = horizontal == 1 ? pos_img_\order2(1, 0, 0, Nx, Ny, Nz)
                                    : pos_img_\order2(0, 1, 0, Nx, Ny, Nz);

  // for through the entire y dimension
  for (int y = 0; y < Nyy; y++) {
    int x_low, x_high;

    // project (0,y) onto detector
    real d = y * o.y + o.z - ss;

    // compute bounds
    x_low = (int)((-s_upper*1.01 - d) * o.w);
    x_high = (int)((s_upper*1.01 - d) * o.w);

    // case the direction is decreasing switch high and low
    if (o.w < (real)0.) {
      int trade = x_low;
      x_low = x_high;
      x_high = trade;
    }

    // make sure x inside image dimensions
    x_low = max(x_low, 0);
    x_high = min(x_high, Nxx - 1);

    // shift position of image depending on horizontal/vertical
    if (horizontal == 1)
      img = img0 + pos_img_\order2(x_low, y, 0, Nx, Ny, Nz);
    if (horizontal == 0)
      img = img0 + pos_img_\order2(y, x_low, 0, Nx, Ny, Nz);

    // integration in x dimension for fixed y
    for (int x = x_low; x <= x_high; x++) {
      // anterpolation weight via normal distance
      real zz = x * o.x + d;

      real weight = ray_weightfkt_\my_variable_type_\order1\order2(
          zz, fabs(o.w) * delta_x / delta_xi, s_under, s_upper,
          difference);

      if (weight > (real)0.) {
        acc += weight * img[0];
      }
      // update image to next position
      img += stride_x;
    }
  }
  // assign value to sinogram
  sino[pos_sino_\order1(s, a, z, Ns, Na, Nz)] =
      acc * delta_x;
}

// ray-driven Radon backprojection
// Computes the backprojection projection in parallel beam geometry for a given
// image. the \my_variable_type_\order1\order2 suffix sets the kernel to the
// suitable precision, contiguity of the sinogram, contiguity of image.
// Input:
// 			img:  Pointer to array representing image (to be computed)
//			      of dimensions Nx times Ny times Nz
//                            (img_shape=Nx times Ny)
//			sino: Pointer to array representing sinogram (to be
//                            transformed) with detector-dimension times
//                            angle-dimension times z dimension
//                      ofs:  Buffer containing geometric informations concerning the
//			      projection-directions (angular information)
//                            Entries are cos, sin, offset and 1/max(|cos|,|sino|)
//			Geometryinformation: Contains various geometric
//                                           information
//                                           [delta_x, delta_xi, Nx, Ny, Ns, Na]
// Output:
//			values inside img are altered to represent the computed
//                      Radon backprojection
__kernel void radon_ray_ad_\my_variable_type_\order1\order2(
    __global real *img, __global real *sino, __constant real8 *ofs,
    __constant real *Geometryinformation) {
  // Extract dimensions
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);
  const int Ns = Geometryinformation[4];
  const int Na = Geometryinformation[5];

  // Extract current position
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);


    // Extract scales
  const real delta_x = Geometryinformation[0];
  const real delta_xi = Geometryinformation[1];


  // Accumulation variable
  real acc = (real)0.;
  real2 c = (real2)(x, y);

  // shift sinogram to correct z-dimension (as this will remain fixed),
  // particularly relevant for "F" contiguity of image
  sino += pos_sino_\order2(0, 0, z, Ns, Na, Nz);

  // Integrate with respect to angular dimension
  for (int a = 0; a < Na; a++) {
    // Extract angular dimensions
    real8 o = ofs[a];

    real s_under = fabs ( fabs(o.x) - fabs(o.y) )/2.;
    real s_upper = fabs ( fabs(o.x) + fabs(o.y) )/2.;
    real difference = s_upper - s_under;

    real Delta_phi = o.s4; // angle_width asociated to the angle

    // compute detector position associated to (x,y) and phi=a
    real s = dot(c, o.s01) + o.s2;

    int s_low = floor(s - s_upper * (real)1.01);
    int s_high = ceil(s + s_upper * (real)1.01);

    s_low = max(s_low, 0);
    s_high = min(s_high, Ns - 1);

    real acc_local = (real)0.;
    for (int p = s_low; p <= s_high; p++) {
      real zz = s - p;
      real weight = ray_weightfkt_\my_variable_type_\order1\order2(
          zz, fabs(o.w) * delta_x / delta_xi, s_under, s_upper,
          difference);
      acc_local +=
          weight * sino[pos_sino_\order2(p, a, 0, Ns, Na, Nz)];
    }

    // accumulate weigthed sum (Delta_Phi weight due to angular resolution)
    acc += Delta_phi * acc_local;
  }

  // Assign value to img
  img[pos_img_\order1(x, y, z, Nx, Ny, Nz)] = acc * delta_xi / delta_x;
}


// Strip Radon Helper functions
real proj_on_interval_\my_variable_type_\order1\order2(real x,real y,real z)
{//Projects the value of z onto the interval [x,y]
return max(x,min(y,z));
}


real strip_central_primative_\my_variable_type_\order1\order2(real t,real kappa, real delta_x)
{//calculates the primative of the strip weight for the central interval


return 1/delta_x * kappa * t ;
}

real strip_left_primative_\my_variable_type_\order1\order2(real t,real kappa, real s_upper, real delta_x,real difference,real inv_difference)
{//calculates the primative of the strip weight for the left interval

if (fabs(difference)<0.0001 * s_upper)
{
return 0;
}
return 1/delta_x * (t*s_upper + t*t/2)*inv_difference*kappa  ;
}

real strip_right_primative_\my_variable_type_\order1\order2(real t,real kappa, real s_upper, real delta_x,real difference,real inv_difference)
{//calculates the primative of the strip weight for the right interval


if (fabs(difference)<0.0001 * s_upper)
{
return 0;
}

return 1/delta_x * (t*s_upper - t*t/2)*inv_difference*kappa  ;
}


real strip_weightfkt_\my_variable_type_\order1\order2(real t,real kappa, real s_under, real s_upper, real difference,real inv_difference,real delta_x)
{//calculates weight function for the strip

real t_central =  proj_on_interval_\my_variable_type_\order1\order2(-s_under,s_under, t);
real t_left =  proj_on_interval_\my_variable_type_\order1\order2(-s_upper,-s_under, t);
real t_right =  proj_on_interval_\my_variable_type_\order1\order2(s_under,s_upper, t);

real val_central = strip_central_primative_\my_variable_type_\order1\order2(t_central, kappa,  delta_x);
real val_left 	 = strip_left_primative_\my_variable_type_\order1\order2(t_left,kappa,  s_upper,  delta_x, difference, inv_difference);
real val_right 	 = strip_right_primative_\my_variable_type_\order1\order2(t_right,kappa,  s_upper,  delta_x, difference, inv_difference);

return val_central+val_left+val_right;
}


// Strip-driven Radon Transform
// Computes the forward projection in parallel beam geometry for a given image.
// the \my_variable_type_\order1\order2 suffix sets the kernel to the suitable
// precision, contiguity of the sinogram, contiguity of image.
// Input:
//			sino: Pointer to array representing sinogram (to be
//                            computed) with detector-dimension times
//                            angle-dimension times z dimension
// 			img:  Pointer to array representing image to be transformed
//			      of dimensions Nx times Ny times Nz
//                            (img_shape=Nx times Ny)
//                      ofs:  Buffer containing geometric informations concerning the
//			      projection-directions (angular information)
//                            Entries are cos, sin, offset and 1/max(|cos|,|sino|)
//			Geometryinformation: Contains various geometric
//                                           information
//                                           [delta_x, delta_xi, Nx, Ny, Ns, Na]
// Output:
//			values inside sino are altered to represent the computed
//                      Radon transform
__kernel void radon_strip_\my_variable_type_\order1\order2(
    __global real *sino, __global real *img, __constant real8 *ofs,
    __constant real *Geometryinformation) {
  // Extract dimensions
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = get_global_size(2);
  const int Nx = Geometryinformation[2];
  const int Ny = Geometryinformation[3];

  // Extract current position
  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = get_global_id(2);

  // Extract scales
  const real delta_x = Geometryinformation[0];
  const real delta_xi = Geometryinformation[1];

  // hack (since otherwise s is unsigned which leads to overflow problems)
  int ss = s;

  // Extract angular information
  // o = (cos,sin,offset,1/max(|cos|,|sin|))
  real4 o = ofs[a].s0123;
  real reverse_mask = ofs[a].s5;
  real s_under = fabs(fabs(o.x) - fabs(o.y)) * (real)0.5;
  real s_upper = (fabs(o.x) + fabs(o.y)) * (real)0.5;
  real difference = s_upper - s_under;
  real inv_difference = (real)1. / difference;

  // Dummy variable for switching from horizontal to vertical lines
  int Nxx = Nx;
  int Nyy = Ny;
  int horizontal = 1;

  // When line is horizontal rather than vertical, switch x and y dimensions
  if (reverse_mask != (real)0.) {
    horizontal = 0;
    o.xy = (real2)(o.y, o.x);

    Nxx = Ny;
    Nyy = Nx;
  }

  // accumulation variable
  real acc = (real)0.;

  // shift image to correct z-dimension (as this will remain fixed),
  // particularly relevant for "F" contiguity of image
  __global real *img0 = img + pos_img_\order2(0, 0, z, Nx, Ny, Nz);

  // stride representing one index_step in x dimension (dependent on
  // horizontal/vertical)
  size_t stride_x = horizontal == 1 ? pos_img_\order2(1, 0, 0, Nx, Ny, Nz)
                                    : pos_img_\order2(0, 1, 0, Nx, Ny, Nz);

  // for through the entire y dimension
  for (int y = 0; y < Nyy; y++) {
    int x_low, x_high;

    // project (0,y) onto detector
    real d = y * o.y + o.z - ss;

    // compute bounds: minimal enclosing integer range (ceil/floor instead of an
    // (int) cast to the enclosing integer). The +-0.5 is the strip box half-width;
    // the *1.01 cushion is kept for boundary robustness. min/max replaces the swap.
    real x_a = (-s_upper*1.01 - 0.5 - d) * o.w;
    real x_b = ( s_upper*1.01 + 0.5 - d) * o.w;
    x_low  = (int)ceil (min(x_a, x_b));
    x_high = (int)floor(max(x_a, x_b));

    // make sure x inside image dimensions
    x_low = max(x_low, 0);
    x_high = min(x_high, Nxx - 1);

    // shift position of image depending on horizontal/vertical
    if (horizontal == 1)
      img = img0 + pos_img_\order2(x_low, y, 0, Nx, Ny, Nz);
    if (horizontal == 0)
      img = img0 + pos_img_\order2(y, x_low, 0, Nx, Ny, Nz);

    // integration in x dimension for fixed y
    for (int x = x_low; x <= x_high; x++) {
      // anterpolation weight via normal distance
      real zz = x * o.x + d;

      real weight1 = strip_weightfkt_\my_variable_type_\order1\order2(
          zz - (real)0.5, fabs(o.w) * delta_x / delta_xi, s_under,
          s_upper, difference, inv_difference, delta_x / delta_xi);
      real weight2 = strip_weightfkt_\my_variable_type_\order1\order2(
          zz + (real)0.5, fabs(o.w) * delta_x / delta_xi, s_under,
          s_upper, difference, inv_difference, delta_x / delta_xi);
      real weight = weight2 - weight1;

      acc += weight * img[0];
      // update image to next position
      img += stride_x;
    }
  }
  // assign value to sinogram
  // delta_x*delta_x/delta_xi (as in the standard/pixel kernel): the strip
  // weight density collapses to |o.w| (the 1/(delta_x/delta_xi) in the strip
  // primitive cancels the delta_x/delta_xi inside kappa), so the extra
  // delta_x/delta_xi factor that the ray weight carries must be supplied here.
  // Without it the strip projection scales as Nx/Ns and only matches ray/pixel
  // on the diagonal Nx==Ns.
  sino[pos_sino_\order1(s, a, z, Ns, Na, Nz)] =
      acc * delta_x * delta_x / delta_xi;
}

// Strip-driven Radon backprojection
// Computes the backprojection projection in parallel beam geometry for a given
// image. the \my_variable_type_\order1\order2 suffix sets the kernel to the
// suitable precision, contiguity of the sinogram, contiguity of image.
// Input:
// 			img:  Pointer to array representing image (to be computed)
//			      of dimensions Nx times Ny times Nz
//                            (img_shape=Nx times Ny)
//			sino: Pointer to array representing sinogram (to be
//                            transformed) with detector-dimension times
//                            angle-dimension times z dimension
//                      ofs:  Buffer containing geometric informations concerning the
//			      projection-directions (angular information)
//                            Entries are cos, sin, offset and 1/max(|cos|,|sino|)
//			Geometryinformation: Contains various geometric
//                                           information
//                                           [delta_x, delta_xi, Nx, Ny, Ns, Na]
// Output:
//			values inside img are altered to represent the computed
//                      Radon backprojection
__kernel void radon_strip_ad_\my_variable_type_\order1\order2(
    __global real *img, __global real *sino, __constant real8 *ofs,
    __constant real *Geometryinformation) {
  // Extract dimensions
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);
  const int Ns = Geometryinformation[4];
  const int Na = Geometryinformation[5];

  // Extract current position
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);


    // Extract scales
  const real delta_x = Geometryinformation[0];
  const real delta_xi = Geometryinformation[1];


  // Accumulation variable
  real acc = (real)0.;
  real2 c = (real2)(x, y);

  // shift sinogram to correct z-dimension (as this will remain fixed),
  // particularly relevant for "F" contiguity of image
  sino += pos_sino_\order2(0, 0, z, Ns, Na, Nz);

  // Integrate with respect to angular dimension
  for (int a = 0; a < Na; a++) {
    // Extract angular dimensions
    real8 o = ofs[a];

    real s_under = fabs(fabs(o.x) - fabs(o.y)) * (real)0.5;
    real s_upper = (fabs(o.x) + fabs(o.y)) * (real)0.5;
    real difference = s_upper - s_under;
    real inv_difference = (real)1. / difference;

    real Delta_phi = o.s4; // angle_width asociated to the angle

    // compute detector position associated to (x,y) and phi=a
    real s = dot(c, o.s01) + o.s2;

    // minimal enclosing integer range (ceil/floor instead of floor/ceil) -> no
    // extra zero-weight detector pixel per side. The +-0.5 is the strip's box
    // half-width; the *1.01 cushion is kept for boundary robustness.
    int s_low = ceil(s-s_upper*1.01-0.5);
    int s_high = floor(s+s_upper*1.01+0.5);

    s_low = max(s_low, 0);
    s_high = min(s_high, Ns - 1);

    real acc_local = (real)0.;
    for (int p = s_low; p <= s_high; p++) {
      real zz = s - p;
      real weight1 = strip_weightfkt_\my_variable_type_\order1\order2(
          zz - (real)0.5, fabs(o.w) * delta_x / delta_xi, s_under,
          s_upper, difference, inv_difference, delta_x / delta_xi);
      real weight2 = strip_weightfkt_\my_variable_type_\order1\order2(
          zz + (real)0.5, fabs(o.w) * delta_x / delta_xi, s_under,
          s_upper, difference, inv_difference, delta_x / delta_xi);
      real weight = weight2 - weight1;
      acc_local +=
          weight * sino[pos_sino_\order2(p, a, 0, Ns, Na, Nz)];
    }


    // accumulate weigthed sum (Delta_Phi weight due to angular resolution)
    acc += Delta_phi * acc_local;
  }

  // Assign value to img
  // analogous to the forward strip fix: the strip weight density collapses to
  // |o.w| (missing the delta_x/delta_xi the ray weight carries), so the extra
  // delta_x/delta_xi factor must be supplied here. Combined with the ray-adjoint
  // scaling delta_xi/delta_x this collapses to a plain acc. Without it the strip
  // backprojection of a constant scales as Nx/Ns and only matches R*(1)=pi on
  // the diagonal Nx==Ns.
  img[pos_img_\order1(x, y, z, Nx, Ny, Nz)] = acc;
}
