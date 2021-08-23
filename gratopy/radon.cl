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
  const float delta_x = Geometryinformation[0];
  const float delta_xi = Geometryinformation[1];

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

  // Extruct current position
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
  const float delta_x = Geometryinformation[0];
  const float delta_xi = Geometryinformation[1];

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
