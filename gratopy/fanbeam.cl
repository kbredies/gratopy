//accessing array contiguous or fortran contiguous
#define pos_img_f(x,y,z,Nx,Ny,Nz) (x+y*Nx+z*Nx*Ny)
#define pos_sino_f(s,a,z,Ns,Na,Nz) (s+a*Ns +Ns*Na*z)
#define pos_img_c(x,y,z,Nx,Ny,Nz) (z+y*Nz+x*Nz*Ny)
#define pos_sino_c(s,a,z,Ns,Na,Nz) (z+a*Nz+s*Nz*Na)




//Fanbeam Transform
// Input:
//			sino: pointer to array representing sinogram (to be computed) with detector-dimension times angle-dimension times z dimension
// 			img:  pointer to array representing image to be transformed of dimens Nx times Ny times Nz (img_shape=Nx times Ny)
//			ofs:  buffer containing geometric informations concerning the projection-directions (angular information)
//				  the first two entries are xd,yd in detetectordirection with length delta_xi
//				  third and fourth (qx,qy) from source to origin with length RE
//				  fith and sixth (dx0,dy0) from origin to center of detector-line
//			sdpd: buffer containing the values associated to sqrt(xi^2+R^2) as weighting
//			Geometry_information:  contains various geometric 
//					0. R 1. RE 2. delta_xi, 3.x_mid,4. y_mid, 5 xi_midpoint, 6. Nx,7.Ny,8.Nxi,9.Nphi,10.delta_x 
// Output:
//			values inside sino are altered to represent the computed fanbeam transform
__kernel void fanbeam_\my_variable_type_\order1\order2(__global \my_variable_type *sino, __global \my_variable_type *img,
                      __constant \my_variable_type8 *ofs, __constant \my_variable_type *sdpd ,
                      __constant \my_variable_type* Geometryinformation)
{	
//Geometric information
size_t Ns = get_global_size(0);
size_t Na = get_global_size(1);
size_t Nz = get_global_size(2);


size_t s = get_global_id(0);
size_t a = get_global_id(1);
size_t z = get_global_id(2);
  
\my_variable_type midpoint_x=Geometryinformation[3];
\my_variable_type midpoint_y=Geometryinformation[4];
\my_variable_type midpoint_det=Geometryinformation[5];

\my_variable_type R= Geometryinformation[0];
\my_variable_type RE= Geometryinformation[1];
\my_variable_type delta_xi= Geometryinformation[2]; //delta_xi / delta_x (so ratio, code runs like delta_x=1)

int Nx=Geometryinformation[6];
int Ny=Geometryinformation[7];

\my_variable_type delta_x = Geometryinformation[10];// True delta_x, i.e. not rescaled like delta_xi

//Geometric information associated with a.th angle
\my_variable_type8 o=ofs[a];
//(xd,yd) ... vector along the detector with length delta_xi.
\my_variable_type xd=o.s0;
\my_variable_type yd=o.s1;
//(qx,qy) ... vector from source to origin (with length RE).
\my_variable_type qx=o.s2;
\my_variable_type qy=o.s3;
//(dx0,dy0) ... vector from  origin orthogonally projected onto detector-line.(with length R-RE)
\my_variable_type dx0=o.s4;
\my_variable_type dy0=o.s5;

//accumulation variable
\my_variable_type acc=0;



// compute direction vector from source to detector pixels above dp (for i+1) and below dm (for i-1).
\my_variable_type dpx=dx0+xd*(-midpoint_det+s+1)-qx;
\my_variable_type dpy=dy0+yd*(-midpoint_det+s+1)-qy;
\my_variable_type dmx=dx0+xd*(-midpoint_det+s-1)-qx;
\my_variable_type dmy=dy0+yd*(-midpoint_det+s-1)-qy;


//direction vector from origin to detector center (orthogonal projection of origin on detector-line)
\my_variable_type dx=dx0-qx;
\my_variable_type dy=dy0-qy;

//Normalization (the norm of (dx,dy) is 1/R after normalization)
dx/=(R*R);
dy/=(R*R);

//Normalization (the norm of (xd,yd) is 1/delta_xi after normalization)
xd/=(delta_xi*delta_xi);
yd/=(delta_xi*delta_xi);

// Distance from source to origin devided by R
RE=RE/R;


int Nyy=Ny;
int Nxx=Nx;


int horizontal;
//Seperate lines which are rather vertical than horizontal 0
//For iteration over y- values for vertical lines have much fewer relevant y
//values with much greater number of corresponding x values, 
// For parallelization it is preferable to iterate over similarly distributed sets
//Therefore, for the vertical case the geometry of the image is flipped,
//so that asside from minor if inquiry all rays execute the same loop
//with similarly distributed iteration-sets
if (fabs(dpx)<fabs(dpy) && fabs(dmx)<fabs(dmy))//mostly horizontal lines
	{horizontal=1;
	}
else //case of vertical lines, switch x and y dimensions of geometry
{horizontal=0;

\my_variable_type trade=dy;
dy=dx;
dx=trade;

trade =yd;
yd= xd;
xd=trade;

trade=qy;
qy=qx;
qx=trade;

trade =dpy;
dpy=dpx;
dpx=trade;

trade =dmy;
dmy=dmx;
dmx=trade;

trade=midpoint_y;
midpoint_y=midpoint_x;
midpoint_x=trade;

Nyy=Nx;
Nxx=Ny;
}

	
//Move in y direction stepwise, rescale dp/dm such that represents increase in y direction
dpx=dpx/dpy;
dmx=dmx/dmy;
	
//compute bounds for suitable x values (for fixed y=0) 
//(qy+midpoint_y distance from source to y=0) according to equation
// x=qx+dmx*(y-qy) with y =-midpoint_y
\my_variable_type xlow=qx-dmx*(qy+midpoint_y);
\my_variable_type xhigh=qx-dpx*(qy+midpoint_y);
	
	
//switch roles of dm and dp if necessary (to switch xlow and xhigh)
if ((qy)*(dmx-dpx)<0)
{
	\my_variable_type trade=xhigh;
	xhigh=xlow;
	xlow=trade;

	trade=dpx;
	dpx=dmx;
	dmx=trade; 
}

//For loop going through all y values
for (int y=0;y<Nyy;y++)
{	
    //changing y by one updates xlow and xhigh exactly by the slopes dp and dm
    // as given in formular above by increasing y by 1 
    xhigh=qx+dpx*(y-midpoint_y-qy);
    xlow=qx+dmx*(y-midpoint_y-qy);

    // cut bounds within image_range
    int xhighint=floor (min(Nxx-1-midpoint_x,xhigh)+midpoint_x);
    int xlowint=ceil(max(-midpoint_x,xlow)+midpoint_x);
    
    
    //alternative stepping
    //xhigh+=dpx;
    //xlow+=dmx;


    // for (x,ylowint) compute t and s orthogonal distances from source (t values in (0,1)) 
    // or projected detectorposition (divided by delta_xi ) 		
    \my_variable_type t=dx*(xlowint-midpoint_x)+dy*(y-midpoint_y)+RE;
    \my_variable_type ss=xd*(xlowint-midpoint_x)+yd*(y-midpoint_y);
    
    
    // loop through all adjacent x values inside the bounds			
    for (int x=xlowint;x<=xhighint;x++)
    {
	// xi is equal the projected detector position (with exact positions as integers) 
	 \my_variable_type xi=ss/t;
	 
	//Weight corresponds to distance of projected detector position 
	//divided by the distance from the source
	\my_variable_type Weight=(1-fabs(s-xi-midpoint_det))/(R*t);
			
	//cut of ray when hits detector (in case detector inside imaging object)
	//if(t>1)
	//{
	//Weight=0;
	//}
	
	//accumulation
	if (horizontal==1)
	{acc+=Weight*img[pos_img_\order2(x,y,z,Nx,Ny,Nz)];}
	
	if(horizontal==0) 
	{//in this case the variable x represents 
	//the true y value and reversely due to flipped geometry, 
	//hence data must be accessed slightly differently
	acc+=Weight*img[pos_img_\order2(y,x,z,Nx,Ny,Nz)];}
			
	//update t and s via obvious formulas (for fixed y) and x increased by 1
	t+=dx;
	ss+=xd;
    }
}

     
//update relevant sinogram value (weighted with spdp=sqrt(xi^2+R^2) 
//(one delta_x is hidden in the R*t term)
//(one delta_xi is hidden in weight with values [0,1] instead of [0,\delta_x])	                 
sino[pos_sino_\order1(s,a,z,Ns,Na,Nz)]=acc*sdpd[s]/delta_xi*delta_x;
}               


 


 
//Fanbeam Backprojection
// Input:
//			img:  pointer to array representing image (to be computed) of dimensions Nx times Ny times Nz (img_shape=Nx times Ny)
// 			sino: pointer to array representing sinogram (to be transformed) with detector-dimension times angle-dimension times z dimension
//			ofs:  buffer containing geometric informations concerning the projection-directions (angular information)
//				  the first two entries are xd,yd in detetectordirection with length delta_xi
//				  third and fourth (qx,qy) from source to origin with length RE
//				  fith and sixth (dx0,dy0) from origin to center of detector-line
//			sdpd: buffer containing the values associated to sqrt(xi^2+R^2) as weighting
//			Geometry_information:  contains various geometric 
//					0. R 1. RE 2. delta_xi, 3.x_mid,4. y_mid, 5 xi_midpoint, 6. Nx,7.Ny,8.Nxi,9.Nphi,10.delta_x 
// Output:
//			values inside img are altered to represent the computed fanbeam backprojection 
__kernel void fanbeam_ad_\my_variable_type_\order1\order2(__global \my_variable_type *img, __global \my_variable_type *sino,
                      __constant \my_variable_type8 *ofs, __constant \my_variable_type *sdpd,  
                      __constant \my_variable_type* Geometryinformation)
{
//geometric information
size_t Nx = get_global_size(0);
size_t Ny = get_global_size(1);
size_t Nz = get_global_size(2);
size_t xx = get_global_id(0);
size_t yy = get_global_id(1);
size_t z = get_global_id(2);

\my_variable_type midpoint_x=Geometryinformation[3];
\my_variable_type midpoint_y=Geometryinformation[4];
\my_variable_type midpoint_det=Geometryinformation[5];

int Ns=Geometryinformation[8];
int Na=Geometryinformation[9];


\my_variable_type delta_xi=Geometryinformation[2];
\my_variable_type R= Geometryinformation[0];

\my_variable_type x=xx-midpoint_x;
\my_variable_type y=yy-midpoint_y;

//accumulation variable
\my_variable_type acc=0;


// for loop through all angles
for (int a=0;a< Na;a++)
{	
	//Geometric information associated with j.th angle
	\my_variable_type8 o=ofs[a];
	//(xd,yd) ... vector along the detector with length delta_xi.
	\my_variable_type xd=o.s0;
	\my_variable_type yd=o.s1;
	//(qx,qy) ... vector from source to origin (with length RE).
	\my_variable_type qx=o.s2;
	\my_variable_type qy=o.s3;
	//(dx0,dy0) ... vector from  origin orthogonally projected onto detector-line (With length R-RE).
	\my_variable_type dx0=o.s4;
	\my_variable_type dy0=o.s5;
	
	//vector from source to detector center (orthogonally project detector-line) with length R
	\my_variable_type dx=dx0-qx;
	\my_variable_type dy=dy0-qy;
	
	//Delta_Phi angular resolution
	\my_variable_type Delta_Phi=o.s6;

	//normalization (afterwards (dx,dy) has norm 1/R).
	dx/=(R*R);
	dy/=(R*R);
	
	//normalization (afterwards (xd,yd) has norm 1/delta_xi).
	xd/=(delta_xi*delta_xi);
	yd/=(delta_xi*delta_xi);
		
	//compute t and s orthogonal distances from source (t values in (0,1)) 
	// or projected detectorposition (divided by delta_xi ) 	
	\my_variable_type t=dx*(x-qx)+dy*(y-qy);
	\my_variable_type ss=xd*(x-qx)+yd*(y-qy);
	
	//compute s projected position on detector
	\my_variable_type xi=ss/t+midpoint_det;
	
	//compute adjacent detector positions 
	int xim=floor(xi);
	int xip=xim+1;
	
	// compute corresponding weights
	\my_variable_type Weightp=1-(xim+1-xi);
	\my_variable_type Weightm=1-(xi-xim);
	
	//set weight to zero in case adjacent detector position is outside 
	//the detector range and weight with corresponding sdpd=sqrt(xi^2+R^2)
	if (xim <0 || xim>Ns-1)
		{Weightm=0.; xim=0;}
	else
		{Weightm*=sdpd[xim];}
	if (xip <0 || xip>Ns-1)
		{Weightp=0.; xip=0;}
	else
		{Weightp*=sdpd[xip];}

	//accumulate weigthed sum (Delta_Phi weight due to angular resolution)
	acc+=Delta_Phi*(Weightm*sino[pos_sino_\order2(xim,a,z,Ns,Na,Nz)]+Weightp*sino[pos_sino_\order2(xip,a,z,Ns,Na,Nz)])/(R*t);
	
}
// update img with computed value
img[pos_img_\order1(xx,yy,z,Nx,Ny,Nz)]=acc;
}





// Single line of Fanbeam Transform: Computes the Fanbeam transform of an image with delta peak in (x,y)
// Input:
//			sino: pointer to array representing sinogram (to be computed) with detector-dimension times angle-dimension times z dimension
//                      x:    x position for which to compute the projection of     
//                      y     y position for which to compute the projection of
//			ofs:  buffer containing geometric informations concerning the projection-directions (angular information)
//				  the first two entries are xd,yd in detetectordirection with length delta_xi
//				  third and fourth (qx,qy) from source to origin with length RE
//				  fith and sixth (dx0,dy0) from origin to center of detector-line
//			sdpd: buffer containing the values associated to sqrt(xi^2+R^2) as weighting
//			Geometry_information:  contains various geometric 
//					0. R 1. RE 2. delta_xi, 3.x_mid,4. y_mid, 5 xi_midpoint, 6. Nx,7.Ny,8.Nxi,9.Nphi,10.delta_x 
// Output:
//			values inside sino are altered to represent the computed fanbeam transform gained by an delta at the position (x,y)
__kernel void single_line_fan_\my_variable_type_\order1\order2(__global \my_variable_type *sino, int x, int y,
                      __constant \my_variable_type8 *ofs, __constant \my_variable_type *sdpd ,
                      __constant \my_variable_type* Geometryinformation)
{	
//Geometric information
size_t Ns = get_global_size(0);
size_t Na = get_global_size(1);
size_t Nz = 1;


size_t s = get_global_id(0);
size_t a = get_global_id(1);
size_t z = 0;
  
\my_variable_type midpoint_x=Geometryinformation[3];
\my_variable_type midpoint_y=Geometryinformation[4];
\my_variable_type midpoint_det=Geometryinformation[5];

\my_variable_type R= Geometryinformation[0];
\my_variable_type RE= Geometryinformation[1];
\my_variable_type delta_xi= Geometryinformation[2]; //delta_xi / delta_x (so ratio, code runs like delta_x=1)

int Nx=Geometryinformation[6];
int Ny=Geometryinformation[7];

\my_variable_type delta_x = Geometryinformation[10];// True delta_x, i.e. not rescaled like delta_xi

//Geometric information associated with a.th angle
\my_variable_type8 o=ofs[a];
//(xd,yd) ... vector along the detector with length delta_xi.
\my_variable_type xd=o.s0;
\my_variable_type yd=o.s1;
//(qx,qy) ... vector from source to origin (with length RE).
\my_variable_type qx=o.s2;
\my_variable_type qy=o.s3;
//(dx0,dy0) ... vector from  origin orthogonally projected onto detector-line.(with length R-RE)
\my_variable_type dx0=o.s4;
\my_variable_type dy0=o.s5;

//accumulation variable
\my_variable_type acc=0;



// compute direction vector from source to detector pixels above dp (for i+1) and below dm (for i-1).
\my_variable_type dpx=dx0+xd*(-midpoint_det+s+1)-qx;
\my_variable_type dpy=dy0+yd*(-midpoint_det+s+1)-qy;
\my_variable_type dmx=dx0+xd*(-midpoint_det+s-1)-qx;
\my_variable_type dmy=dy0+yd*(-midpoint_det+s-1)-qy;


//direction vector from origin to detector center (orthogonal projection of origin on detector-line)
\my_variable_type dx=dx0-qx;
\my_variable_type dy=dy0-qy;

//Normalization (the norm of (dx,dy) is 1/R after normalization)
dx/=(R*R);
dy/=(R*R);

//Normalization (the norm of (xd,yd) is 1/delta_xi after normalization)
xd/=(delta_xi*delta_xi);
yd/=(delta_xi*delta_xi);

// Distance from source to origin devided by R
RE=RE/R;


int Nyy=Ny;
int Nxx=Nx;


int horizontal;
//Seperate lines which are rather vertical than horizontal 0
//For iteration over y- values for vertical lines have much fewer relevant y
//values with much greater number of corresponding x values, 
// For parallelization it is preferable to iterate over similarly distributed sets
//Therefore, for the vertical case the geometry of the image is flipped,
//so that asside from minor if inquiry all rays execute the same loop
//with similarly distributed iteration-sets
if (fabs(dpx)<fabs(dpy) && fabs(dmx)<fabs(dmy))//mostly horizontal lines
	{horizontal=1;
	}
else //case of vertical lines, switch x and y dimensions of geometry
{horizontal=0;

\my_variable_type trade=dy;
dy=dx;
dx=trade;

trade =yd;
yd= xd;
xd=trade;

trade=qy;
qy=qx;
qx=trade;

trade =dpy;
dpy=dpx;
dpx=trade;

trade =dmy;
dmy=dmx;
dmx=trade;

trade=midpoint_y;
midpoint_y=midpoint_x;
midpoint_x=trade;

Nyy=Nx;
Nxx=Ny;

trade=y;
y=x;
x=trade;
}

	
//Move in y direction stepwise, rescale dp/dm such that represents increase in y direction
dpx=dpx/dpy;
dmx=dmx/dmy;
	
//compute bounds for suitable x values (for fixed y=0) 
//(qy+midpoint_y distance from source to y=0) according to equation
// x=qx+dmx*(y-qy) with y =-midpoint_y
\my_variable_type xlow=qx-dmx*(qy+midpoint_y);
\my_variable_type xhigh=qx-dpx*(qy+midpoint_y);
	
	
//switch roles of dm and dp if necessary (to switch xlow and xhigh)
if ((qy)*(dmx-dpx)<0)
{
	\my_variable_type trade=xhigh;
	xhigh=xlow;
	xlow=trade;

	trade=dpx;
	dpx=dmx;
	dmx=trade; 
}

    //Compute xlow and xhigh for given y
    xhigh=qx+dpx*(y-midpoint_y-qy);
    xlow=qx+dmx*(y-midpoint_y-qy);

    // cut bounds within image_range
    int xhighint=floor (min(Nxx-1-midpoint_x,xhigh)+midpoint_x);
    int xlowint=ceil(max(-midpoint_x,xlow)+midpoint_x);
    
    
    //alternative stepping
    //xhigh+=dpx;
    //xlow+=dmx;


    // for (x,y) compute t and s orthogonal distances from source (t values in (0,1)) 
    // or projected detectorposition (divided by delta_xi ) in case x is in feasible range		
    if ((xlowint<=x) && (x<=xhighint))
    {
    \my_variable_type t=dx*(x-midpoint_x)+dy*(y-midpoint_y)+RE;
    \my_variable_type ss=xd*(x-midpoint_x)+yd*(y-midpoint_y);
    
    // xi is equal the projected detector position (with exact positions as integers) 
     \my_variable_type xi=ss/t;
     
    //Weight corresponds to distance of projected detector position 
    //divided by the distance from the source
    \my_variable_type Weight=(1-fabs(s-xi-midpoint_det))/(R*t);
		    
    //cut of ray when hits detector (in case detector inside imaging object)
    //if(t>1)
    //{
    //Weight=0;
    //}
    
    //accumulation
    if (horizontal==1)
    {acc+=Weight*1;}
    
    if(horizontal==0) 
    {//in this case the variable x represents 
    //the true y value and reversely due to flipped geometry, 
    //hence data must be accessed slightly differently
    acc+=Weight*1;}
		    
    }


     
//update relevant sinogram value (weighted with spdp=sqrt(xi^2+R^2) 
//(one delta_x is hidden in the R*t term)
//(one delta_xi is hidden in weight with values [0,1] instead of [0,\delta_x])	                 
sino[pos_sino_\order1(s,a,z,Ns,Na,Nz)]=acc*sdpd[s]/delta_xi*delta_x;
}     
