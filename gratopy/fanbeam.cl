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
#undef real8
#endif
#define real \my_variable_type
#define real2 \my_variable_type2
#define real8 \my_variable_type8

// Fanbeam transform
// Computes the forward projection in fanbeam geometry for a given image.
// the \my_variable_type_\order1\order2 suffix sets the kernel to the suitable
// precision, contiguity of the sinogram, contiguity of image.
// Input:
//			sino: Pointer to array representing sinogram (to be
//                            computed) with detector-dimension times angle-dimension
//                            times z dimension
// 			img:  Pointer to array representing image to be transformed of
//                            dimensions Nx times Ny times Nz (img_shape=Nx times Ny)
//			ofs:  Buffer containing geometric informations concerning
//                            the projection directions (angular information),
//                            The first two entries are (xd,yd) in direction along the
//                            detector line with length delta_xi.
//                            Third and fourth entries (qx,qy) from origin to source with
//                            length RE.
//                            Fifth and sixth entries (dx0,dy0) from origin to center of
//                            detector line (orthogonal projection of origin onto
//                            detector line).
//                      sdpd: Buffer containing the values associated with sqrt(xi^2+R^2)
//                            as weighting
//     Geometry_information:  Contains various geometric quantities
//                            relevant for the computation, more precisely
//                            0. R/delta_x, 1. RE/delta_x, 2. delta_xi/delta_x,
//                            3. x_mid (in image_pixels), 4. y_mid (in image_pixels),
//                            5. xi_midpoint (in detector_pixels), 6. Nx, 7. Ny, 8. Nxi,
//                            9. Nphi, 10. delta_x
// Output:
//			values inside sino are altered to represent the computed
//                      fanbeam transform
__kernel void fanbeam_\my_variable_type_\order1\order2(
    __global real *sino, __global real *img, __constant real8 *ofs,
    __constant real *sdpd, __constant real *Geometryinformation) {
  // Extract geometric information
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = get_global_size(2);
  int Nx = Geometryinformation[6];
  int Ny = Geometryinformation[7];

  // Extract current position
  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = get_global_id(2);

  // Midpoints of geometry
  real2 midpoint = (real2)(Geometryinformation[3], Geometryinformation[4]);
  real midpoint_det = Geometryinformation[5];

  // Relevant distances
  real R = Geometryinformation[0];
  real RE = Geometryinformation[1];
  real delta_xi = Geometryinformation[2]; // delta_xi / delta_x (so ratio, code
                                          // runs like delta_x=1)
  real delta_x =
      Geometryinformation[10]; // True delta_x, i.e. not rescaled like delta_xi

  // Geometric information associated with a.th angle
  real8 o = ofs[a];
  //(xd,yd) ... vector along the detector-line with length delta_xi.
  real2 d = (real2)(o.s0, o.s1);
  //(qx,qy) ... vector from origin to source (with length RE).
  real2 q = (real2)(o.s2, o.s3);
  //(dx0,dy0) ... vector from  origin orthogonally projected onto
  //detector-line.(with length R-RE)
  real2 d0 = (real2)(o.s4, o.s5);

  // accumulation variable
  real acc = (real)0.;

  // compute direction vector from source to detector pixels above dp (for i+1)
  // and below dm (for i-1).
  real2 dp = d0 + d * (-midpoint_det + s + 1) - q;
  // (real2)(d0.x + d.x*(-midpoint_det+s+1)-q.x,
  // d0.y+d.y*(-midpoint_det+s+1)-q.y);
  real2 dm = d0 + d * (-midpoint_det + s - 1) - q;
  // dm = (rea	l2)(d0.x+d.x*(-midpoint_det+s-1)-q.x,
  // d0.y+d.y*(-midpoint_det+s-1)-q.y);

  // direction vector from origin to detector center (orthogonal projection of
  // origin on detector-line)
  real2 dd = d0 - q;

  // Normalization (the norm of dd is 1/R after normalization)
  dd /= R * R;

  // Normalization (the norm of d is 1/delta_xi after normalization)
  d /= delta_xi * delta_xi;

  // Distance from source to origin divided by R
  // RE=RE/R; Bug for Intel double precision
  RE *= 1 / R;

  // Dummy variable in case dimensions are switched
  int Nyy = Ny;
  int Nxx = Nx;
  int horizontal;

  // Seperate lines which are rather vertical than horizontal 0
  // For iteration over y- values for vertical lines have much fewer relevant y
  // values with much greater number of corresponding x values,
  // For parallelization it is preferable to iterate over similarly distributed
  // sets Therefore, for the vertical case the geometry of the image is flipped,
  // so that asside from minor if inquiry all rays execute the same loop
  // with similarly distributed iteration-sets.
  if (fabs(dp.x) < fabs(dp.y) &&
      fabs(dm.x) < fabs(dm.y)) { // mostly horizontal lines
    horizontal = 1;
  } else { // case of vertical lines, switch x and y dimensions of geometry
    horizontal = 0;

    dd = (real2)(dd.y, dd.x);
    d = (real2)(d.y, d.x);
    q = (real2)(q.y, q.x);
    dp = (real2)(dp.y, dp.x);
    dm = (real2)(dm.y, dm.x);
    midpoint = (real2)(midpoint.y, midpoint.x);

    Nyy = Nx;
    Nxx = Ny;
  }

  // Move in y direction stepwise, rescale dp/dm such that represents increase
  // in y direction
  dp.x /= dp.y;
  dm.x /= dm.y;

  // compute bounds for suitable x values (for fixed y=0)
  //(qy+midpoint_y distance from source to y=0) according to equation
  // x=qx+dmx*(y-qy) with y =-midpoint_y
  real xlow = q.x - dm.x * (q.y + midpoint.y);
  real xhigh = q.x - dp.x * (q.y + midpoint.y);

  // switch roles of dm and dp if necessary (to switch xlow and xhigh)
  if ((q.y) * (dm.x - dp.x) < (real)0.) {
    real trade = xhigh;
    xhigh = xlow;
    xlow = trade;

    trade = dp.x;
    dp.x = dm.x;
    dm.x = trade;
  }

  // detector position s relative to detector midpoint
  real s_midpoint_det = s - midpoint_det;

  // shift image to correct z-dimension (as this will remain fixed),
  // particularly relevant for "F" contiguity of imate
  __global real *img0 = img + pos_img_\order2(0, 0, z, Nx, Ny, Nz);

  // stride representing one index_step in x dimension (dependent on
  // horizontal/vertical)
  size_t stride_x = (horizontal != 0) ? pos_img_\order2(1, 0, 0, Nx, Ny, Nz)
                                      : pos_img_\order2(0, 1, 0, Nx, Ny, Nz);

  // For loop going through all y values
  for (int y = 0; y < Nyy; y++) {
    // changing y by one updates xlow and xhigh exactly by the slopes dp and dm
    // as given in formular above by increasing y by 1
    xhigh = q.x + dp.x * (y - midpoint.y - q.y);
    xlow = q.x + dm.x * (y - midpoint.y - q.y);

    // cut bounds within image_range
    int xhighint = floor(min(Nxx - 1 - midpoint.x, xhigh) + midpoint.x);
    int xlowint = ceil(max(-midpoint.x, xlow) + midpoint.x);

    // alternative stepping (might lead to numercal issues for large images)
    // xhigh+=dpx;
    // xlow+=dmx;

    // for (x,ylowint) compute t and ss orthogonal distances from source (t
    // values in (0,1)) and projected detectorposition (divided by delta_xi )
    real t = dd.x * (xlowint - midpoint.x) + dd.y * (y - midpoint.y) + RE;
    real ss = d.x * (xlowint - midpoint.x) + d.y * (y - midpoint.y);

    // shift image to relevant positions
    if (horizontal != 0)
      img = img0 + pos_img_\order2(xlowint, y, 0, Nx, Ny, Nz);
    if (horizontal == 0)
      img = img0 + pos_img_\order2(y, xlowint, 0, Nx, Ny, Nz);

    // loop through all adjacent x values inside the bounds
    for (int x = xlowint; x <= xhighint; x++) {
      // xi is equal the projected detector position (with exact positions as
      // integers)
      real xi = ss / t;

      // Weight corresponds to distance of projected detector position
      // divided by the distance from the source
      real Weight = ((real)1. - fabs(s_midpoint_det - xi)) / t;

      // cut of ray when hits detector (in case detector inside imaging object)
      // if(t>1)
      //{
      // Weight=0;
      //}

      // accumulation of values
      acc += Weight * img[0];
      // update image to next position
      img += stride_x;

      // update t and s via obvious formulas (for fixed y) and x increased by 1
      t += dd.x;
      ss += d.x;
    }
  }

  // update relevant sinogram value (weighted with spdp=sqrt(xi^2+R^2)
  //(one delta_x is hidden in the R*t term)
  //(one delta_xi is hidden in weight with values [0,1] instead of [0,\delta_x])
  sino[pos_sino_\order1(s, a, z, Ns, Na, Nz)] =
      acc * sdpd[s] / delta_xi * delta_x / R;
}

// Fanbeam backprojection
// Computes the backprojection projection in fanbeam geometry for a given image.
// the \my_variable_type_\order1\order2 suffix sets the kernel to the suitable
// precision, contiguity of the image, contiguity of sinogram.
// Input:
// 			img:  Pointer to array representing image (to be computed) of
//                            dimensions Nx times Ny times Nz (img_shape=Nx times Ny)
//			sino: Pointer to array representing sinogram (to be
//                            transformed) with detector-dimension times angle-dimension
//                            times z dimension
//			ofs:  Buffer containing geometric informations concerning
//                            the projection directions (angular information),
//                            The first two entries are (xd,yd) in direction along the
//                            detector line with length delta_xi.
//                            Third and fourth entries (qx,qy) from origin to source with
//                            length RE.
//                            Fifth and sixth entries (dx0,dy0) from origin to center of
//                            detector line (orthogonal projection of origin onto
//                            detector line).
//                      sdpd: Buffer containing the values associated with sqrt(xi^2+R^2)
//                            as weighting
//     Geometry_information:  Contains various geometric quantities
//                            relevant for the computation, more precisely
//                            0. R/delta_x, 1. RE/delta_x, 2. delta_xi/delta_x,
//                            3. x_mid (in image_pixels), 4. y_mid (in image_pixels),
//                            5. xi_midpoint (in detector_pixels), 6. Nx, 7. Ny, 8. Nxi,
//                            9. Nphi, 10. delta_x
// Output:
//			values inside img are altered to represent the computed
//                      fanbeam backprojection


__kernel void fanbeam_ad_\my_variable_type_\order1\order2(
    __global real *img, __global real *sino, __constant real8 *ofs,
    __constant real *sdpd, __constant real *Geometryinformation) {
  // Extract geometric information
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);
  int Ns = Geometryinformation[8];
  int Na = Geometryinformation[9];

  // Extract current position
  size_t xx = get_global_id(0);
  size_t yy = get_global_id(1);
  size_t z = get_global_id(2);

  // Midpoints of the geometry
  real2 midpoint = (real2)(Geometryinformation[3], Geometryinformation[4]);
  real midpoint_det = Geometryinformation[5];

  // Relevant distances
  real delta_xi = Geometryinformation[2];
  real R = Geometryinformation[0];

  // P shift image position relative to image midpoint
  real2 P = (real2)(xx, yy) - midpoint;

  // accumulation variable
  real acc = (real)0.;

  // Shift sinogram to suitable z-position
  sino += pos_sino_\order2(0, 0, z, Ns, Na, Nz);

  // precompute scaling parameters
  real R_sqr_inv = (real)1. / (R * R);
  real delta_xi_sqr_inv = (real)1. / (delta_xi * delta_xi);

  // for loop through all angles
  for (int a = 0; a < Na; a++) {
    // Geometric information associated with a.th angle
    real8 o = ofs[a];
    //(xd,yd) ... vector along the detector with length delta_xi.
    real2 d = (real2)(o.s0, o.s1);
    //(qx,qy) ... vector from origin to source (with length RE).
    real2 q = (real2)(o.s2, o.s3);
    //(dx0,dy0) ... vector from  origin orthogonally projected onto
    //detector-line.(with length R-RE)
    real2 d0 = (real2)(o.s4, o.s5);
    // direction vector from origin to detector center (orthogonal projection of
    // origin on detector-line)
    real2 dd = d0 - q;

    // Delta_Phi angular resolution
    real Delta_Phi = o.s6;

    // normalization (afterwards (dx,dy) has norm 1/R).
    dd *= R_sqr_inv;

    // normalization (afterwards (xd,yd) has norm 1/delta_xi).
    d *= delta_xi_sqr_inv;

    // compute t and s orthogonal distances from source (t values in (0,1))
    // or projected detectorposition (divided by delta_xi )
    // real t = dd.x*(P.x-q.x)+dd.y*(P.y-q.y);
    // real ss = d.x*(P.x-q.x)+d.y*(P.y-q.y);
    real t = dot(dd, P - q);
    real ss = dot(d, P - q);

    // compute xi projected position on detector
    real xi = ss / t + midpoint_det;

    // compute adjacent detector positions
    int xim = floor(xi);
    int xip = xim + 1;

    // compute corresponding weights
    real Weightp = (real)1. - (xip - xi);
    real Weightm = (real)1. - (xi - xim);

    // set weight to zero in case adjacent detector position is outside
    // the detector range and weight with corresponding sdpd=sqrt(xi^2+R^2)
    if (xim < 0 || xim >= Ns) {
      Weightm = (real)0.;
      xim = 0;
    } else {
      Weightm *= sdpd[xim];
    }
    if (xip < 0 || xip >= Ns) {
      Weightp = (real)0.;
      xip = 0;
    } else {
      Weightp *= sdpd[xip];
    }

    // accumulate weigthed sum (Delta_Phi weight due to angular resolution)
    acc += Delta_Phi *
           (Weightm * sino[pos_sino_\order2(xim, a, 0, Ns, Na, Nz)] +
            Weightp * sino[pos_sino_\order2(xip, a, 0, Ns, Na, Nz)]) /
           (R * t);
  }
  // update img with computed value
  img[pos_img_\order1(xx, yy, z, Nx, Ny, Nz)] = acc;
}

// Single line of Fanbeam Transform: Computes the Fanbeam transform of an image
// with delta peak in (x,y)
// Input:
//			sino: Pointer to array representing sinogram (to be
//                            computed) with detector-dimension times angle-dimension
//                            times z dimension
// 			x,y:  Position of the delta peak to be transformed
//			ofs:  Buffer containing geometric informations concerning
//                            the projection directions (angular information),
//                            The first two entries are (xd,yd) in direction along the
//                            detector line with length delta_xi.
//                            Third and fourth entries (qx,qy) from origin to source with
//                            length RE.
//                            Fifth and sixth entries (dx0,dy0) from origin to center of
//                            detector line (orthogonal projection of origin onto
//                            detector line).
//                      sdpd: Buffer containing the values associated with sqrt(xi^2+R^2)
//                            as weighting
//     Geometry_information:  Contains various geometric quantities
//                            relevant for the computation, more precisely
//                            0. R/delta_x, 1. RE/delta_x, 2. delta_xi/delta_x,
//                            3. x_mid (in image_pixels), 4. y_mid (in image_pixels),
//                            5. xi_midpoint (in detector_pixels), 6. Nx, 7. Ny, 8. Nxi,
//                            9. Nphi, 10. delta_x
// Output:
//			values inside sino are altered to represent the computed
//                      fanbeam transform obtained by transforming an image with
//                      Dirac-delta at (x,y)

__kernel void single_line_fan_\my_variable_type_\order1\order2(
    __global real *sino, int x, int y, __constant real8 *ofs,
    __constant real *sdpd, __constant real *Geometryinformation) {
  // Extract geometric sizes
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1);
  size_t Nz = 1;
  int Nx = Geometryinformation[6];
  int Ny = Geometryinformation[7];

  // Extract current position
  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = 0;

  // Extract Geometry midpoints
  real2 midpoint = (real2)(Geometryinformation[3], Geometryinformation[4]);
  real midpoint_det = Geometryinformation[5];

  // Relevant distacnes
  real R = Geometryinformation[0];
  real RE = Geometryinformation[1];
  real delta_xi = Geometryinformation[2]; // delta_xi / delta_x (so ratio, code
                                          // runs like delta_x=1)
  real delta_x =
      Geometryinformation[10]; // True delta_x, i.e. not rescaled like delta_xi

  // Geometric information associated with a.th angle
  real8 o = ofs[a];
  //(xd,yd) ... vector along the detector with length delta_xi.
  real2 d = (real2)(o.s0, o.s1);
  //(qx,qy) ... vector from origin to source (with length RE).
  real2 q = (real2)(o.s2, o.s3);
  //(dx0,dy0) ... vector from  origin orthogonally projected onto
  //detector-line.(with length R-RE)
  real2 d0 = (real2)(o.s4, o.s5);

  // accumulation variable
  real acc = 0;

  // compute direction vector from source to detector pixels above dp (for i+1)
  // and below dm (for i-1).
  real2 dp = d0 + d * (-midpoint_det + s + 1) - q;
  real2 dm = d0 + d * (-midpoint_det + s - 1) - q;

  // direction vector from origin to detector center (orthogonal projection of
  // origin on detector-line)
  real2 dd = d0 - q;

  // Normalization (the norm of dd is 1/R after normalization)
  dd /= R * R;

  // Normalization (the norm of d is 1/delta_xi after normalization)
  d /= delta_xi * delta_xi;

  // Distance from source to origin divided by R
  // RE=RE/R; Intel bug for Double
  RE *= 1 / R;

  int Nyy = Ny;
  int Nxx = Nx;

  int horizontal;
  // Seperate lines which are rather vertical than horizontal 0
  // For iteration over y- values for vertical lines have much fewer relevant y
  // values with much greater number of corresponding x values,
  // For parallelization it is preferable to iterate over similarly distributed
  // sets
  // Therefore, for the vertical case the geometry of the image is flipped,
  // so that asside from minor if inquiry all rays execute the same loop
  // with similarly distributed iteration-sets
  if (fabs(dp.x) < fabs(dp.y) &&
      fabs(dm.x) < fabs(dm.y)) { // mostly horizontal lines
    horizontal = 1;
  } else { // case of vertical lines, switch x and y dimensions of geometry
    horizontal = 0;

    dd = (real2)(dd.y, dd.x);
    d = (real2)(d.y, d.x);
    q = (real2)(q.y, q.x);
    dp = (real2)(dp.y, dp.x);
    dm = (real2)(dm.y, dm.x);
    midpoint = (real2)(midpoint.y, midpoint.x);

    Nyy = Nx;
    Nxx = Ny;

    real trade = y;
    y = x;
    x = trade;
  }

  // Move in y direction stepwise, rescale dp/dm such that represents increase
  // in y direction
  dp.x /= dp.y;
  dm.x /= dm.y;

  // compute bounds for suitable x values (for fixed y=0)
  //(qy+midpoint_y distance from source to y=0) according to equation
  // x=qx+dmx*(y-qy) with y =-midpoint_y
  real xlow = q.x - dm.x * (q.y + midpoint.y);
  real xhigh = q.x - dp.x * (q.y + midpoint.y);

  // switch roles of dm and dp if necessary (to switch xlow and xhigh)
  if ((q.y) * (dm.x - dp.x) < (real)0.) {
    real trade = xhigh;
    xhigh = xlow;
    xlow = trade;

    trade = dp.x;
    dp.x = dm.x;
    dm.x = trade;
  }

  // Compute xlow and xhigh for given y
  xhigh = q.x + dp.x * (y - midpoint.y - q.y);
  xlow = q.x + dm.x * (y - midpoint.y - q.y);

  // cut bounds within image_range
  int xhighint = floor(min(Nxx - 1 - midpoint.x, xhigh) + midpoint.x);
  int xlowint = ceil(max(-midpoint.x, xlow) + midpoint.x);

  // for (x,y) compute t and s orthogonal distances from source (t values in
  // (0,1)) or projected detectorposition (divided by delta_xi ) in case x is in
  // feasible range
  if ((xlowint <= x) && (x <= xhighint)) {
    real t = dd.x * (x - midpoint.x) + dd.y * (y - midpoint.y) + RE;
    real ss = d.x * (x - midpoint.x) + d.y * (y - midpoint.y);

    // xi is equal the projected detector position (with exact positions as
    // integers)
    real xi = ss / t;

    // Weight corresponds to distance of projected detector position
    // divided by the distance from the source
    real Weight = ((real)1. - fabs(s - xi - midpoint_det)) / (R * t);

    // accumulation
    acc = Weight;
  }

  // update relevant sinogram value (weighted with spdp=sqrt(xi^2+R^2)
  //(one delta_x is hidden in the R*t term)
  //(one delta_xi is hidden in weight with values [0,1] instead of [0,\delta_x])
  sino[pos_sino_\order1(s, a, z, Ns, Na, Nz)] =
      acc * sdpd[s] / delta_xi * delta_x;
}
