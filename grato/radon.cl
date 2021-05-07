
//accessing array contiguous or fortran contiguous
#define pos_img_f(x,y,z,Nx,Ny,Nz) (x+y*Nx+z*Nx*Ny)
#define pos_sino_f(s,a,z,Ns,Na,Nz) (s+a*Ns +Ns*Na*z)
#define pos_img_c(x,y,z,Nx,Ny,Nz) (z+y*Nz+x*Nz*Ny)
#define pos_sino_c(s,a,z,Ns,Na,Nz) (z+a*Nz+s*Nz*Na)




//Radon Transform
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
__kernel void radon_\my_variable_type_\order1\order2(__global \my_variable_type *sino,
					__global \my_variable_type *img,__constant \my_variable_type8 *ofs,
					 __constant \my_variable_type* Geometryinformation)
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
  
  //hack (since otherwise i is unsigned which leads to overflow problems)
  int ss=s;
  
   
  //o = (cos,sin,offset,1/cos)
  \my_variable_type4 o = ofs[a].s0123;
  
  //accumulation variable
  \my_variable_type acc = 0.0f;

  //for through the entire y dimension
  for(int y = 0; y < Ny; y++) {
  
	int x_low, x_high;
	
	//project (0,y) onto detector
	\my_variable_type d = y*o.y + o.z;

	// compute bounds
	if (o.x == 0) { //case ray is horizontal there are only two detectors hit
	  if ((d > ss-1) && (d < ss+1)) {
		x_low = 0; x_high = Nx-1;
	  } else { // horizontal ray, but detector in question is not hit, jump to next y
		 continue;
	  }
	} else if (o.x > 0) {//ray increases in x dimension
	  x_low = (int)((ss-1 - d)*o.w);
	  x_high = (int)((ss+1 - d)*o.w);
	} else {//ray decreasing in x dimension
	  x_low = (int)((ss+1 - d)*o.w);
	  x_high = (int)((ss-1 - d)*o.w);
	}
	
	//make sure x inside image dimensions
	x_low = max(x_low, 0);
	x_high = min(x_high, Nx-1);

	// integration in x dimension for fixed y
	for(int x = x_low; x <= x_high; x++) {
	  //anterpolation weight via normal distance
	  \my_variable_type weight = 1.0 - fabs(x*o.x + d - ss);
	  if (weight > 0.0f) 
		acc += weight*img[pos_img_\order2(x,y,z,Nx,Ny,Nz)];
	}
  }
  //assign value to sinogram
  sino[ pos_sino_\order1(s,a,z,Ns,Na,Nz)] = acc*delta_x*delta_x/delta_xi;
}


//Fanbeam Backprojection
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
__kernel void radon_ad_\my_variable_type_\order1\order2(__global \my_variable_type *img, __global \my_variable_type *sino,
					   __constant \my_variable_type8 *ofs, 
					   __constant \my_variable_type* Geometryinformation
					   )
{
  //Geometric and discretization information
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz = get_global_size(2);
  
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);
  
  const int Ns = Geometryinformation[4];
  const int Na = Geometryinformation[5];


  //accumulation variable
  \my_variable_type acc = 0.0f;


  \my_variable_type4 c = (\my_variable_type4)(x,y,1,0);

  
  //integrate with respect to angular dimension
  for (int a=0; a < Na; a++) {
    
    \my_variable_type Delta_phi=ofs[a].s4; //angle_width asociated to the angle
	
	//compute detector position associated to (x,y) and phi=a
	\my_variable_type s = dot(c, ofs[a].s0123);
	
	//make sure detector position is inside range
	if ((s > -1) && (s < Ns)) {
	  \my_variable_type s_floor;
	  \my_variable_type p = fract(s, &s_floor);
	  if (s_floor >= 0)	  acc += Delta_phi*(1.0f - p)*sino[ pos_sino_\order2((int)s_floor,a,z,Ns,Na,Nz)];
	  if (s_floor <= Ns-2) acc += Delta_phi*p*sino[ pos_sino_\order2((int)(s_floor+1),a,z,Ns,Na,Nz)];
	}
  }
  //assign value to sinogram
  img[pos_img_\order1(x,y,z,Nx,Ny,Nz)] = acc;
}


