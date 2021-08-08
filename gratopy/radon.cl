// Array indexing for C contiguous or Fortran contiguous arrays
#define pos_img_f(x,y,z,Nx,Ny,Nz) (x+Nx*(y+Ny*z))
#define pos_sino_f(s,a,z,Ns,Na,Nz) (s+Ns*(a+Na*z))
#define pos_img_c(x,y,z,Nx,Ny,Nz) (z+Nz*(y+Ny*x))
#define pos_sino_c(s,a,z,Ns,Na,Nz) (z+Nz*(a+Na*s))

#ifdef real
#undef real
#undef real2
#undef real4
#undef real8
#endif
#define real \my_variable_type
#define real2 \my_variable_type2
#define real4 \my_variable_type4
#define real8 \my_variable_type8 

// Radon Transform
// Input:
//			sino: pointer to array representing sinogram (to be computed) 
//				  with detector-dimension times angle-dimension times z dimension
// 			img:  pointer to array representing image to be transformed of
//				  dimens Nx times Ny times Nz (img_shape=Nx times Ny)
//			ofs:  buffer containing geometric informations concerning the 
//					projection-directions (angular information)
//				  Entries are cos, sin, ofset and 1/cos
//			Geometry_information:  contains various geometric information 
//					[delta_x,delta_xi,Nx,Ny,Ni,Nj] 
// Output:
//			values inside sino are altered to represent the computed Radon transform	
__kernel void radon_\my_variable_type_\order1\order2(__global real *sino,
						     __global real *img,
						     __constant real8 *ofs,
						     __constant real* Geometryinformation)
{
  //Geometric and discretization information
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1); 
  size_t Nz = get_global_size(2);
  
  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = get_global_id(2);
  
  const int Nx= Geometryinformation[2];
  const int Ny= Geometryinformation[3];
  
  const float delta_x= Geometryinformation[0];
  const float delta_xi= Geometryinformation[1];
  
  //hack (since otherwise s is unsigned which leads to overflow problems)
  int ss=s;
  
  //o = (cos,sin,offset,1/cos)
  real4 o = ofs[a].s0123;
  
  int Nxx=Nx;
  int Nyy=Ny;

  int horizontal=1;
  if (fabs(o.x)<=fabs(o.y))
    {
      horizontal=0;

      o.xy = (real2)(o.y,o.x);
      
      Nxx=Ny;
      Nyy=Nx;
    }
  
  // accumulation variable
  real acc = 0.0f;

  __global real *img0 = img + pos_img_\order2(0,0,z,Nx,Ny,Nz);
  size_t stride_x = horizontal == 1 ? pos_img_\order2(1,0,0,Nx,Ny,Nz) : pos_img_\order2(0,1,0,Nx,Ny,Nz);

  // for through the entire y dimension
  for(int y = 0; y < Nyy; y++) {
    int x_low, x_high;
    
    //project (0,y) onto detector
    real d = y*o.y + o.z;
    
    // compute bounds
    x_low = (int)((ss-1 - d)*o.w);
    x_high = (int)((ss+1 - d)*o.w);
    
    if (o.w<0)
      {
	int trade = x_low;
	x_low = x_high;
	x_high=trade;
      }
    
    //make sure x inside image dimensions
    x_low = max(x_low, 0);
    x_high = min(x_high, Nxx-1);
    
    if (horizontal == 1)
      img = img0 + pos_img_\order2(x_low,y,0,Nx,Ny,Nz);
    if (horizontal == 0)
      img = img0 + pos_img_\order2(y,x_low,0,Nx,Ny,Nz);
    
    // integration in x dimension for fixed y
    for(int x = x_low; x <= x_high; x++) {
      //anterpolation weight via normal distance
      real weight = 1.0 - fabs(x*o.x + d - ss);
      if (weight > 0.0f) {
	acc += weight*img[0];
      }
      img += stride_x;
    }
  }
  //assign value to sinogram
  sino[pos_sino_\order1(s,a,z,Ns,Na,Nz)] = acc*delta_x*delta_x/delta_xi;
}

// Radon backprojection
// Input:
//			img:  pointer to array representing image (to be computed) of dimensions Nx times Ny times Nz (img_shape=Nx times Ny)
// 			sino: pointer to array representing sinogram (to be transformed) with detector-dimension times angle-dimension times z dimension
//			ofs:  buffer containing geometric informations concerning the 
//				  projection-directions (angular information)
//				  Entries are cos, sin, ofset and 1/cos
//	//			Geometry_information:  contains various geometric information 
//					[delta_x,delta_xi,Nx,Ny,Ni,Nj] 
// Output:
//			values inside img are altered to represent the computed Radon backprojection 
__kernel void radon_ad_\my_variable_type_\order1\order2(__global real *img,
							__global real *sino,
							__constant real8 *ofs, 
							__constant real* Geometryinformation
							)
{
  // Geometric and discretization information
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);
  
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);
  
  const int Ns = Geometryinformation[4];
  const int Na = Geometryinformation[5];

  // Accumulation variable
  real acc = 0.0f;
  real4 c = (real4)(x,y,1,0);

  sino += pos_sino_\order2(0,0,z,Ns,Na,Nz);  
  // Integrate with respect to angular dimension
  for (int a=0; a < Na; a++) {
    real Delta_phi=ofs[a].s4; //angle_width asociated to the angle
    
    //compute detector position associated to (x,y) and phi=a
    real s = dot(c, ofs[a].s0123);
    
    //make sure detector position is inside range
    if ((s > -1) && (s < Ns)) {
      real s_f;
      real p = fract(s, &s_f);
      int s_floor = (int)s_f;
      if (s_floor >= 0)	  acc += Delta_phi*(1.0f - p)*sino[pos_sino_\order2(s_floor,a,0,Ns,Na,Nz)];
      s_floor++;
      if (s_floor < Ns) acc += Delta_phi*p*sino[ pos_sino_\order2(s_floor,a,0,Ns,Na,Nz)];
    } 
  }
  // Assign value to img
  img[pos_img_\order1(x,y,z,Nx,Ny,Nz)] = acc;
  
}

// Single Line of Radon Transform: Computes the Fanbeam transform of an image with delta peak in (x,y) 
// Input:
//			sino: pointer to array representing sinogram (to be computed) 
//				  with detector-dimension times angle-dimension times z dimension
// 			img:  pointer to array representing image to be transformed of
//				  dimens Nx times Ny times Nz (img_shape=Nx times Ny)
//			ofs:  buffer containing geometric informations concerning the 
//					projection-directions (angular information)
//				  Entries are cos, sin, ofset and 1/cos
//			Geometry_information:  contains various geometric information 
//					[delta_x,delta_xi,Nx,Ny,Ni,Nj] 
// Output:
//			values inside sino are altered to represent the computed Radon transform
//                      obtained by transforming an image with dirac-delta at (x,y)	
__kernel void single_line_radon_\my_variable_type_\order1\order2(__global real *sino,
								 int x,  int y,
								 __constant real8 *ofs,
								 __constant real* Geometryinformation)
{  //Geometric and discretization information
  size_t Ns = get_global_size(0);
  size_t Na = get_global_size(1); 
  size_t Nz = 1;
  
  size_t s = get_global_id(0);
  size_t a = get_global_id(1);
  size_t z = 0;
  
  const int Nx= Geometryinformation[2];
  const int Ny= Geometryinformation[3];
  
  const float delta_x= Geometryinformation[0];
  const float delta_xi= Geometryinformation[1];
  
  // hack (since otherwise s is unsigned which leads to overflow problems)
  int ss=s;
  
  //o = (cos,sin,offset,1/cos)
  real4 o = ofs[a].s0123;
  
  int Nxx=Nx;
  int Nyy=Ny;
  
  int horizontal=1;
  if (fabs(o.x)<=fabs(o.y))
    {
      horizontal=0;

      o.xy = (real2)(o.y, o.x);
      
      Nxx=Ny;
      Nyy=Nx;
      
      real trade=x;
      x=y;
      y=trade;
    }
  
  //accumulation variable
  real acc = 0.0f;
  
  // for through the entire y dimension
  int x_low, x_high;
  
  //project (0,y) onto detector
  real d = y*o.y + o.z;
  
  // compute bounds
  x_low = (int)((ss-1 - d)*o.w);
  x_high = (int)((ss+1 - d)*o.w);
  
  if (o.w<0)
    {
      int trade = x_low;
      x_low = x_high;
      x_high=trade;
    }
  
  //make sure x inside image dimensions
  x_low = max(x_low, 0);
  x_high = min(x_high, Nxx-1);
  
  // integration in x dimension for fixed y
  if((x_low<=x) && (x<=x_high)){
    //anterpolation weight via normal distance
    real weight = 1.0 - fabs(x*o.x + d - ss);
    if (weight > 0.0f) {
      acc = weight;
    }
  }
  
  //assign value to sinogram
  sino[ pos_sino_\order1(s,a,z,Ns,Na,Nz)] = acc*delta_x*delta_x/delta_xi;
}


