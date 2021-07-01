#define pos_img_f(x,y,z,Nx,Ny,Nz) (x+y*Nx+z*Nx*Ny)
#define pos_sino_f(s,a,z,Ns,Na,Nz) (s+a*Ns +Ns*Na*z)
#define pos_img_c(x,y,z,Nx,Ny,Nz) (z+y*Nz+x*Nz*Ny)
#define pos_sino_c(s,a,z,Ns,Na,Nz) (z+a*Nz+s*Nz*Na)


  __kernel void empty_test_\my_variable_type_\order1\order2()
  {
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz=get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);

  size_t i =pos_img_\order1(x,y,z,Nx,Ny,Nz);
  }

  

  __kernel void update_v_\my_variable_type_\order1\order2(__global \my_variable_type3 *v, __global \my_variable_type *u,
					   const float sigma, const float z_distance) {
		 
  size_t Nx = get_global_size(0);
  size_t Ny = get_global_size(1);
  size_t Nz=get_global_size(2);
  size_t asdf=get_global_size(1);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);

  size_t i =pos_img_\order1(x,y,z,Nx,Ny,Nz);
  
  int xx=x+1;
  int yy=y+1;
  int zz=z+1;



  // gradient 
  \my_variable_type3 val = -u[i];
  if (x < Nx-1) val.s0 += u[pos_img_\order1(xx,y,z,Nx,Ny,Nz)];
    else val.s0 = 0.0f;
  if (y < Ny-1) val.s1 += u[pos_img_\order1(x,yy,z,Nx,Ny,Nz)];
   else val.s1 = 0.0f;
  if (z < Nz-1)	  val.s2 += u[pos_img_\order1(x,y,zz,Nx,Ny,Nz)];
    else val.s2=0.0f;
  
  if (z_distance>0){
    val.s2/=z_distance; //adjust to further Jump
    }
  else{
  val.s2=0;
  }
  // step
  v[i] = v[i] + sigma*(val);
  }
  
  
  __kernel void update_lambda_L2_\my_variable_type_\order1\order2(__global \my_variable_type *lambda, __global \my_variable_type *Ku,__global \my_variable_type *f, const float sigma,
							const float sigmap1inv) {
		 
  size_t Ns = get_global_size(0), Na = get_global_size(1), Nz=get_global_size(2);
  size_t s = get_global_id(0), a = get_global_id(1);
  size_t z=get_global_id(2);
  size_t i =pos_sino_\order1(s,a,z,Ns,Na,Nz);


  lambda[i] = (lambda[i] + sigma*(Ku[i] - f[i]))*sigmap1inv;
}

__kernel void update_u_\my_variable_type_\order1\order2(__global \my_variable_type *u, __global \my_variable_type *u_,
					   __global \my_variable_type3 *v, __global \my_variable_type *Kstarlambda,
					   const float tau, const float norming, const float z_distance) {
		 
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1);
  size_t z=get_global_id(2);
  size_t i = pos_img_\order1(x,y,z,Nx,Ny,Nz);
  
    

  int xx=x-1;
  int yy= y-1;
  int zz=z-1;
 
  
  // divergence
  \my_variable_type3 val = v[i];
  
  if (x == Nx-1) val.s0 = 0.0f;
  if (x > 0) val.s0 -= v[pos_img_\order1(xx,y,z,Nx,Ny,Nz)].s0;  
  
  if (y == Ny-1) val.s1 = 0.0f;
  if (y > 0) val.s1 -= v[pos_img_\order1(x,yy,z,Nx,Ny,Nz)].s1;
  
  if (z == Nz-1) val.s2 = 0.0f;
  if (z > 0) val.s2-=v[pos_img_\order1(x,y,z-1,Nx,Ny,Nz)].s2;
  if (z_distance>0) val.s2/=z_distance; //adjust fur further step


  


  // linear step
  u[i] = u_[i] + tau*(val.s0 + val.s1 + val.s2 - norming*Kstarlambda[i]);
  if(u[i]<0){u[i]=0.;}
}






__kernel void update_NormV_unchor_\my_variable_type_\order1\order2(__global \my_variable_type3 *V,__global \my_variable_type *normV) {
  
  size_t Nx = get_global_size(0), Ny = get_global_size(1), Nz=get_global_size(2);
  size_t x = get_global_id(0), y = get_global_id(1),z=get_global_id(2);
  
  size_t i = pos_img_\order1(x,y,z,Nx,Ny,Nz);

	//Computing Norm
	\my_variable_type norm=0;	
	norm= hypot(hypot(V[i].s0,V[i].s1),V[i].s2);
	if (norm > 1.0f) {
	V[i]/=norm;}
	}
	
    
