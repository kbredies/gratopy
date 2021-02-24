from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import pyopencl.array as clarray
import scipy.misc
import time
from fanbeam_source import *


####################
## fan beam CPU code

    

def fanbeam_struct_richy_cpu( shape, angles, detector_width,
                   source_detector_dist, source_origin_dist,
                   n_detectors=None, detector_shift = 0.0):
	
	if isscalar(angles):
		angles = linspace(0,2*pi,angles+1)[:-1] + pi

	image_pixels = min(shape)
	if n_detectors is None:
		nd = image_pixels
	else:
		nd = n_detectors

	midpoint_domain = array([shape[0]-1, shape[1]-1])/2.0
	midpoint_detectors = (nd-1.0)/2.0
	
	
	#find suitable imagedimensions encapsulated by the rays
	b=(detector_width-1*detector_width/nd)/2.
	beta=np.arctan(abs(b/source_detector_dist))
	r=sin(beta)*abs(source_origin_dist)
	r=min(r,abs(source_detector_dist-source_origin_dist))
	r=r*(1/(1+2./nd))
	image_width=2**0.5*r

	

	# adjust distances to pixel units
	#image_width = detector_width*source_origin_dist/float(source_detector_dist)/(2**0.5)
	source_detector_dist *= image_pixels/image_width
	source_origin_dist *= image_pixels/image_width
	detector_width *= image_pixels/image_width

	# offset function parameters
	thetaX = -cos(angles)
	thetaY = sin(angles)
	XD=thetaX*detector_width/nd
	YD=thetaY*detector_width/nd

	Qx=-thetaY*source_origin_dist
	Qy=thetaX*source_origin_dist

	Dx0= thetaY*(source_detector_dist-source_origin_dist)
	Dy0= -thetaX*(source_detector_dist-source_origin_dist)

	
	Overallangle, differencealpha, Sinvalues=compute_angles(nd,midpoint_detectors,detector_width/nd /source_detector_dist)
	
	#ofs = zeros((8, len(angles)), dtype=float32, order='F')
	#ofs[0,:] = thetaX; ofs[1,:] = thetaY
	#ofs[2,:] = XD; ofs[3,:] = YD
	#ofs[4,:]=Qx; ofs[5,:]=Qy
	#ofs[6,:]=Dx; ofs[7]=Dy

	#ofs_buf = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY, len(ofs.data))
	#cl.enqueue_copy(queue, ofs_buf, ofs.data).wait()

	sinogram_shape = (nd, len(angles))

	return (shape,sinogram_shape,thetaX,thetaY,XD,YD,Qx,Qy,Dx0,Dy0,Overallangle,differencealpha,Sinvalues)
	
	
	



	
	

def compute_angles(N,c,ratio):
	alpha=[]
	Sin=[]
	Alpha=[]
	for u in range(0,N+2):
		s=u-1-c
		s=s*ratio
		
		alpha.append(np.arctan(s))
	for u in range(0,N+1):
		 Alpha.append(alpha[u+1]-alpha[u])
	#Alpha.append(0)
	for i in range(len(Alpha)):
		Sin.append(np.sin(Alpha[i]))
	return alpha, Alpha, Sin
	
def fanbeam_richy_cpu(img,fanbeam_struct):
	shape,sinogram_shape,thetaX,thetaY,XD,YD,Qx,Qy,Dx0,Dy0,Overallangle,differencealpha,Sinvalues=fanbeam_struct
	Epsilon=0.000
	c=(sinogram_shape[0]-1)/2.
	d1=(shape[0]-1)/2
	d2=(shape[1]-1)/2
	Sumofweights=[]
	result=np.zeros(sinogram_shape)
	for j in range( sinogram_shape[1]):
		print j
		Sums=0
		X=thetaX[j];Y=thetaY[j];Xd=XD[j]; Yd=YD[j];qx=Qx[j]; qy=Qy[j];dx0=Dx0[j];dy0=Dy0[j]; 
		for i in range(sinogram_shape[0]):
			
			img2=np.ones(shape)*0.1
			dx=dx0+(i-c)*Xd
			dy=dy0+(i-c)*Yd
			
			dx=dx-qx
			dy=dy-qy
			norm=(dx**2+dy**2)**0.5
			dx/=norm; dy/=norm
			
			wx=dy
			wy=-dx
			
		
			ofset=qx*wx+qy*wy
			ofsetperp=-qx*dx-qy*dy
			
			wpx=wx-dx*Sinvalues[i+1]; wpy=wy-dy*Sinvalues[i+1]
			wmx=-wx-dx*Sinvalues[i]; wmy=-wy-dy*Sinvalues[i]
			
			
			if wpx*wmx<-Epsilon:
				for y in range(shape[1]):
					xp=1/wpx*(-wpy*(y-d2)+ofset+Sinvalues[i+1]*ofsetperp)
					xm=1/wmx*(-wmy*(y-d2)-ofset+Sinvalues[i]*ofsetperp)
					xlow=int(math.ceil( max(-d1,min(xp,xm))+d1))
					xhigh=int(math.floor( min(d1,max(xp,xm))+d1))
					
					for x in range(xlow,xhigh+1):
						delta=wx*(x-d1)+wy*(y-d2)-ofset
						if delta>0:
							R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i+1]
							
						else:
							R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i]
							
						delta=abs(delta)
						#import pdb; pdb.set_trace()
						weight=(R-delta)/R
						Sums+=weight
						result[i,j]+=weight*img[x,y]
						img2[x,y]=weight
			elif wpx*wmx>Epsilon and wmx>0:
				for y in range(shape[1]):
					xp=1/wpx*(-wpy*(y-d2)+ofset+Sinvalues[i+1]*ofsetperp)
					xm=1/wmx*(-wmy*(y-d2)-ofset+Sinvalues[i]*ofsetperp)
					xlow=0
					xhigh=int(math.floor( min(d1,min(xp,xm))+d1))
					#import pdb; pdb.set_trace()
					for x in range(xlow,xhigh+1):
						delta=wx*(x-d1)+wy*(y-d2)-ofset
						if delta>0:
							R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i+1]
							
						else:
							R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i]
							
						delta=abs(delta)
						#import pdb; pdb.set_trace()
						weight=(R-delta)/R

						result[i,j]+=weight*img[x,y]
						img2[x,y]=weight
				
			elif wpx*wmx>Epsilon and wmx<0:
				for y in range(shape[1]):
					xp=1/wpx*(-wpy*(y-d2)+ofset+Sinvalues[i+1]*ofsetperp)
					xm=1/wmx*(-wmy*(y-d2)-ofset+Sinvalues[i]*ofsetperp)
					xlow=int(math.ceil( max(-d1,max(xp,xm))+d1))
					xhigh=2*d1
					
					for x in range(xlow,xhigh+1):
						delta=wx*(x-d1)+wy*(y-d2)-ofset
						if delta>0:
							R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i+1]
							
						else:
							R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i]
							
						delta=abs(delta)
						#import pdb; pdb.set_trace()
						weight=(R-delta)/R

						result[i,j]+=weight*img[x,y]
						img2[x,y]=weight	
			elif abs(wpx*wmx)<=Epsilon and wpy*wmy<0:
				for x in range(shape[0]):
					yp=1/wpy*(-wpx*(x-d1)+ofset+Sinvalues[i+1]*ofsetperp)
					ym=1/wmy*(-wmx*(x-d1)-ofset+Sinvalues[i]*ofsetperp)
					ylow=int(math.ceil( max(-d2,min(yp,ym))+d2))
					yhigh=int(math.floor( min(d2,max(yp,ym))+d2))
					
					for y in range(ylow,yhigh+1):
						delta=wx*(x-d1)+wy*(y-d2)-ofset
						if delta>0:
							R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i+1]
							
						else:
							R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i]
							
						delta=abs(delta)
						
						weight=(R-delta)/R
						Sums+=weight
						result[i,j]+=weight*img[x,y]
						#result[i,j]+=1000
						img2[x,y]=weight	
				
				
			else:
				print 'Something strange going on, topolgical error'
				import pdb; pdb.set_trace()
				raise ValueError
		
			
			from scipy import misc
			
			#misc.imshow(img2)
			#misc.imsave('result'+str(count)+'.png',img2)
			
		Sumofweights.append(Sums)
	#import pdb; pdb.set_trace()
	return result
    

	
def fanbeam_richy_cpu_add(sino,fanbeam_struct):
	
	shape,sinogram_shape,thetaX,thetaY,XD,YD,Qx,Qy,Dx0,Dy0,Overallangle,differencealpha,Sinvalues=fanbeam_struct
	Nx=shape[0];Ny=shape[1]
	
	c=(sinogram_shape[0]-1)/2.
	d1=(Nx-1)/2.; d2=(Ny-1)/2.
	image=np.zeros(shape)
	for xx in range(0,Nx):
		print xx
		x=xx-d1
		for yy in range(0,Ny):
			y=yy-d2
			for j in range(sinogram_shape[1]):
				qx=Qx[j];qy=Qy[j];dx0=Dx0[j]; dy0=Dy0[j]; xd=XD[j]; yd=YD[j]
				zx=x-qx; zy=y-qy;
				a=dx0-qx
				
				if zx==0:
					s=-a/xd
				else:
					b=dy0-qy
					s=(b-a/zx*zy)/(-yd+xd*zy/zx)
				
				
				sm=int(math.floor(s+c))
				sp=sm+1

				dpx=dx0-qx+(sp-c)*xd; dpy=dy0-qy+(sp-c)*yd
				dmx=dx0-qx+(sm-c)*xd; dmy=dy0-qy+(sm-c)*yd
				
				normp=hypot(dpx,dpy)
				normm=hypot(dmx,dmy)
				dpx/=normp; dpy/=normp; dmx/=normm; dmy/=normm
				
				wpx=-dpy; wpy=dpx; wmx=dmy;wmy=-dmx

				deltam=abs(wmx*zx+wmy*zy); deltap=abs(wpx*zx+wpy*zy)
				Rm=Sinvalues[sp]*(dmy*zy+dmx*zx); Rp=Sinvalues[sp]*(dpx*zx+dmy*zy)
				Weightp=(Rp-deltap)/Rp; Weightm=(Rm-deltam)/Rm
				
				if sm<=0:
					Weightm=0
					sm=0
				if sp>=sinogram_shape[0]-1:
					sp=0
					Weightp=0
				
				if Weightp<0 or Weightm<0:
					import pdb; pdb.set_trace()
					
				image[xx,yy]+= Weightp*sino[sp,j]+Weightm*sino[sm,j]

					


	return image
	
	
def fanbeam_richy_cpu_add_individual(x,y,j,sino,fanbeam_struct):
	
	shape,sinogram_shape,thetaX,thetaY,XD,YD,Qx,Qy,Dx0,Dy0,Overallangle,differencealpha,Sinvalues=fanbeam_struct
	Nx=shape[0];Ny=shape[1]
	
	c=(sinogram_shape[0]-1)/2.
	d1=(Nx-1)/2.; d2=(Ny-1)/2.
	image=np.zeros(shape)
	#x=x-d1
	#y=y-d2
	X=thetaX[j];Y=thetaY[j];xd=XD[j]; yd=YD[j];qx=Qx[j]; qy=Qy[j];dx0=Dx0[j];dy0=Dy0[j]; 
	dx=dx0-qx
	dy=dy0-qy
	
	norm=hypot(dx,dy)
	dx/=norm**2
	dy/=norm**2
	
	norm=hypot(xd,yd)
	xd/=norm**2
	yd/=norm**2
	a=dx*(x-d1-qx)+dy*(y-d2-qy)
	b=xd*(x-d1-qx)+yd*(y-d2-qy)
	s=b/a+c
	sm=int(floor(s))
	sp=int(sm+1)
	Weightp=1-(sm+1-s)
	Weightm=1-(s-sm)
	#import pdb; pdb.set_trace()
	if sm not in range(0,sinogram_shape[0]):
		Weightm=0; sm=0;
		
	if sp not in range(0,sinogram_shape[0]):
		Weightp=0; sp=0
		
	print sp, sm
	#import pdb; pdb.set_trace()
	return Weightp,Weightm,sm
	
def fanbeam_richy_cpu_individual(i,j,x,y,img,fanbeam_struct):
	shape,sinogram_shape,thetaX,thetaY,XD,YD,Qx,Qy,Dx0,Dy0,Overallangle,differencealpha,Sinvalues=fanbeam_struct
	Epsilon=0.000
	c=(sinogram_shape[0]-1)/2.
	d1=(shape[0]-1)/2
	d2=(shape[1]-1)/2
	Sumofweights=[]
	result=np.zeros(sinogram_shape)
	
		
	Sums=0
	X=thetaX[j];Y=thetaY[j];Xd=XD[j]; Yd=YD[j];qx=Qx[j]; qy=Qy[j];dx0=Dx0[j];dy0=Dy0[j]; 
					
	img2=np.ones(shape)*0.1
	dx=dx0+(i-c)*Xd
	dy=dy0+(i-c)*Yd
	
	dx=dx-qx
	dy=dy-qy
	norm=(dx**2+dy**2)**0.5
	dx/=norm; dy/=norm
	
	wx=dy
	wy=-dx
	
	
	ofset=qx*wx+qy*wy
	ofsetperp=-qx*dx-qy*dy
	
	wpx=wx-dx*Sinvalues[i+1]; wpy=wy-dy*Sinvalues[i+1]
	wmx=-wx-dx*Sinvalues[i]; wmy=-wy-dy*Sinvalues[i]


	weight=-1

	xp=1/wpx*(-wpy*(y-d2)+ofset+Sinvalues[i+1]*ofsetperp)
	xm=1/wmx*(-wmy*(y-d2)-ofset+Sinvalues[i]*ofsetperp)
	xlow=int(math.ceil( max(-d1,min(xp,xm))+d1))
	xhigh=int(math.floor( min(d1,max(xp,xm))+d1))
	
	if x in range(xlow,xhigh+1):
		delta=wx*(x-d1)+wy*(y-d2)-ofset
		if delta>0:
			R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i+1]
			
		else:
			R=(dx*(x-d1)+dy*(y-d2)+ofsetperp)*Sinvalues[i]
			
		delta=abs(delta)
		#import pdb; pdb.set_trace()
		weight=(R-delta)/R
		Sums+=weight
		result[i,j]+=weight*img[x,y]
		img2[x,y]=weight

	

	
	from scipy import misc
	
	#misc.imshow(img2)
	#misc.imsave('result'+str(count)+'.png',img2)
		
	Sumofweights.append(Sums)
	#import pdb; pdb.set_trace()
	return weight



def radon_normest(queue, r_struct):
	
	img = clarray.to_device(queue, require((random.randn(*r_struct[0])), float32, 'F'))
	
	sino = clarray.zeros(queue, r_struct[1], dtype=float32, order='F')

	V=(fanbeam_richy_gpu(sino, img, r_struct, wait_for=img.events))

	for i in range(50):
		#normsqr = float(sum(img.get()**2)**0.5)
		normsqr = float(clarray.sum(img).get())
		img /= normsqr
		#import pdb; pdb.set_trace()
		sino.events.append( fanbeam_richy_gpu (sino, img, r_struct, wait_for=img.events))
		img.events.append(fanbeam_richy_gpu_add(img, sino, r_struct, wait_for=sino.events))
		
		if i%10==0:
			print 'normest',i, normsqr
	return sqrt(normsqr)


	
	
def fanbeam_add_new(sino,fanbeam_struct):
	shape,sinogram_shape,thetaX,thetaY,XD,YD,Qx,Qy,Dx0,Dy0,Overallangle,differencealpha,Sinvalues=fanbeam_struct
	Epsilon=0.000
	c=(sinogram_shape[0]-1)/2.
	d1=(shape[0]-1)/2
	d2=(shape[1]-1)/2
	result=np.zeros(shape)
	
	for y in range(0,shape[1]):
		for x in range(0,shape[0]):
			
			for j in range(0,sinogram_shape[1]):
				X=thetaX[j];Y=thetaY[j];xd=XD[j]; yd=YD[j];qx=Qx[j]; qy=Qy[j];dx0=Dx0[j];dy0=Dy0[j]; 
				
				
				X=thetaX[j];Y=thetaY[j];xd=XD[j]; yd=YD[j];qx=Qx[j]; qy=Qy[j];dx0=Dx0[j];dy0=Dy0[j]; 
				dx=dx0-qx
				dy=dy0-qy
				
				norm=hypot(dx,dy)
				dx/=norm**2
				dy/=norm**2
				
				norm=hypot(xd,yd)
				xd/=norm**2
				yd/=norm**2
				a=dx*(x-d1-qx)+dy*(y-d2-qy)
				b=xd*(x-d1-qx)+yd*(y-d2-qy)
				s=b/a+c
				sm=int(floor(s))
				sp=int(sm+1)
				Weightp=1-(sm+1-s)
				Weightm=1-(s-sm)
				#import pdb; pdb.set_trace()
				if sm not in range(0,sinogram_shape[0]):
					Weightm=0; sm=0;
				if sp not in range(0,sinogram_shape[0]):
					Weightp=0; sp=0
				
				result[x,y]+=(Weightm)*sino[sm,j]+(Weightp)*sino[sp,j]
	from scipy import misc
	misc.imshow(result)
				
	
def fanbeam_new(img,fanbeam_struct):
	
	
	shape,sinogram_shape,thetaX,thetaY,XD,YD,Qx,Qy,Dx0,Dy0,Overallangle,differencealpha,Sinvalues=fanbeam_struct
	Epsilon=0.000
	c=(sinogram_shape[0]-1)/2.
	d1=(shape[0]-1)/2
	d2=(shape[1]-1)/2
	result=np.zeros(sinogram_shape)
	

	for j in range(sinogram_shape[1]):		
		for i in range(0,int(2*c+1)):
			X=thetaX[j];Y=thetaY[j];xd=XD[j]; yd=YD[j];qx=Qx[j]; qy=Qy[j];dx0=Dx0[j];dy0=Dy0[j]; 
			img2=np.ones(shape)*0.1
			img2[0,0]=0
			
			
			
			dpx=dx0+xd*(i+1-c)-qx;
			dpy=dy0+yd*(i+1-c)-qy;

			dmx=dx0+xd*(i-1-c)-qx;
			dmy=dy0+yd*(i-c-1)-qy;
			
			e=abs(dpx)<abs(dpy) and abs(dmx)<abs(dmy)
			if abs(dpx)<abs(dpy) and abs(dmx)<abs(dmy):	
				dpx=dpx/dpy
				dmx=dmx/dmy
				
				dx=dx0-qx
				dy=dy0-qy
				
				norm=hypot(dx,dy)
				dx/=norm**2
				dy/=norm**2
				
				norm=hypot(xd,yd)
				xd/=norm**2
				yd/=norm**2

				xlow=qx-dmx*(qy+d2);
				xhigh=qx-dpx*(qy+d2);
				#import pdb; pdb.set_trace()

				if xlow>xhigh:
					trade=xhigh;
					xhigh=xlow;
					xlow=trade;

					trade=dpx;
					dpx=dmx;
					dmx=trade; 
				#print i,j

				for y in range(0,shape[1]):
						
					xhighint=int(floor (min(shape[1]-d1-1,xhigh)+d1));
					xlowint=int(ceil(max(-d1,xlow)+d1));

					xhigh+=dpx
					xlow+=dmx
					a=dx*(xlowint-d1-qx)+dy*(y-d2-qy)
					b=xd*(xlowint-d1-qx)+yd*(y-d2-qy)
					
					
					
					for x in range(xlowint,xhighint):
						
						S=b/a
						a+=dx;
						b+=xd;
						Weight=1-abs(i-S-c)
						if Weight<0:
							misc.imshow(img2)
							import pdb; pdb.set_trace()
							print 'Achtung, Weight=',Weight
							
						img2[x,y]=1
						result[i,j]+=Weight*img[x,y]
						
						#import pdb; pdb.set_trace()
				from scipy import misc
			else:
				#print i,j

				dpy=dpy/dpx
				dmy=dmy/dmx
				
				dx=dx0-qx
				dy=dy0-qy
				
				norm=hypot(dx,dy)
				dx/=norm**2
				dy/=norm**2
				
				norm=hypot(xd,yd)
				xd/=norm**2
				yd/=norm**2
				
				ylow=qy-dmy*(qx+d1);
				yhigh=qy-dpy*(qx+d1);
		
				
				if ylow>yhigh:
					trade=yhigh;
					yhigh=ylow;
					ylow=trade;

					trade=dpy;
					dpy=dmy;
					dmy=trade; 
				if i==15 and j==30:
					print 'xbounds',ylow,yhigh
					print 'q', qx,qy
					print 'dp',dpx ,dpy,'dmx',dmx,dmy
				for x in range(0,shape[0]):
						
					yhighint=int(floor (min(shape[1]-1-d2,yhigh)+d2));
					ylowint=int(ceil(max(-d2,ylow)+d2));
					yhigh+=dpy
					ylow+=dmy
					a=dy*(ylowint-d2-qy)+dx*(x-d1-qx)
					b=yd*(ylowint-d2-qy)+xd*(x-d1-qx)
					
					for y in range(ylowint,yhighint):	
						S=b/a
						a+=dy;
						b+=yd;
						Weight=1-abs(i-S-c)
						if Weight<0:
							misc.imshow(img2)
							import pdb; pdb.set_trace()
							print 'Achtung, Weight=',Weight
							
						img2[x,y]=1
						result[i,j]+=Weight*img[x,y]
					
					
#	misc.imshow(result)
	#import pdb; pdb.set_trace()
	return result
	

# test
if __name__ == '__main__':
	
	
	img=np.zeros([225,225,2])
	A =  imread('brain.png')[:,:,0]
	A/=np.max(A)
	B=imread('Ente.jpeg')[:,:,0]
	B=np.array(B,dtype=float)
	B/=np.max(B)
	
	img[:,:,0]=A
	img[:,:,1] = B
	#from scipy import misc
	#misc.imshow(img[:,:,1])
	
	img=np.array(img,dtype=float)
	number_detectors = 512
	#from scipy import misc;misc.imshow(img)
	angles=360
		
	f_struct_gpu = fanbeam_struct_richy_gpu(img.shape, angles,  114.8, 700, 350, number_detectors,0,None)
		
	img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
	import pdb;pdb.set_trace()
	sino_gpu = clarray.zeros(queue, (f_struct_gpu[1][0],f_struct_gpu[1][1],2), dtype=float32, order='F')
	
	a=time.clock()
	for i in range(100):
		sino_gpu.events.append(fanbeam_richy_gpu(sino_gpu,img_gpu,f_struct_gpu,wait_for=sino_gpu.events))
		
	print 'Time Required Forward',(time.clock()-a)/100
	#from scipy import misc;misc.imshow(sino_gpu.get())
	shape,sinogram_shape,ofs_buf,Geometryinfo_buf=f_struct_gpu
	#import pdb; pdb.set_trace()

	#exit
	sino=sino_gpu.get()
	
	numberofangles=180
	angles = linspace(0,2*pi,numberofangles+1)[:-1] + pi
	
	
	#misc.imshow(sino_gpu.get())
	print 'jetz'
	a=time.clock()
	for i in range(100):
		img_gpu.events.append(fanbeam_richy_gpu_add(img_gpu,sino_gpu,f_struct_gpu,wait_for=img_gpu.events))
	print 'Time Required Backprojection',(time.clock()-a)/100
	#misc.imshow(img_gpu.get())
	figure(1)
	imshow(hstack([img[:,:,0],img[:,:,1]]), cmap=cm.gray)
	title('original image')
	figure(2)
	imshow(hstack([sino_gpu.get()[:,:,0],sino_gpu.get()[:,:,1]]), cmap=cm.gray)
	title('Fanbeam transformed image')
	figure(3)
	imshow(hstack([img_gpu.get()[:,:,0],img_gpu.get()[:,:,1]]), cmap=cm.gray)
	title('Backprojected image')
	show()
	import pdb;pdb.set_trace()

	
	
	
	
	img = imread('brain.png')
	img=img[:,:,0]
	#img=np.ones(img.shape)
	img=np.array(img,dtype=float)
	number_detectors = 512
	#from scipy import misc;misc.imshow(img)
	angles=360
	
	f_struct = fanbeam_struct_richy_cpu(img.shape, angles, 114.8, 700, 350, number_detectors)
	#fanbeam_add_new(sino,f_struct)
	
	f_struct_gpu = fanbeam_struct_richy_gpu(img.shape, angles,  114.8, 700, 350, number_detectors,0,None)
	show_geometry(np.pi/4,f_struct_gpu)
	
	#sinonew=fanbeam_new(img,f_struct)
	#sinocpu=fanbeam_richy_cpu(img,f_struct)
	#import pdb; pdb.set_trace()

	#misc.imshow(sino)
	img_gpu = clarray.to_device(queue, require(img, float32, 'F'))
	sino_gpu = clarray.zeros(queue, f_struct_gpu[1], dtype=float32, order='F')
	
	a=time.clock()
	for i in range(100):
		sino_gpu.events.append(fanbeam_richy_gpu(sino_gpu,img_gpu,f_struct_gpu,wait_for=sino_gpu.events))
		
	print 'Time Required Forward',(time.clock()-a)/100
	#from scipy import misc;misc.imshow(sino_gpu.get())
	shape,sinogram_shape,ofs_buf,Geometryinfo_buf=f_struct_gpu
	#import pdb; pdb.set_trace()

	#exit
	sino=sino_gpu.get()
	
	numberofangles=180
	angles = linspace(0,2*pi,numberofangles+1)[:-1] + pi
	
	
	#misc.imshow(sino_gpu.get())
	print 'jetz'
	a=time.clock()
	for i in range(100):
		img_gpu.events.append(fanbeam_richy_gpu_add(img_gpu,sino_gpu,f_struct_gpu,wait_for=img_gpu.events))
	print 'Time Required Backprojection',(time.clock()-a)/100
	#misc.imshow(img_gpu.get())
	figure(1)
	imshow(img, cmap=cm.gray)
	title('original image')
	figure(2)
	imshow(sino_gpu.get(), cmap=cm.gray)
	title('Fanbeam transformed image')
	figure(3)
	imshow(img_gpu.get(), cmap=cm.gray)
	title('Backprojected image')
	show()
	import pdb;pdb.set_trace()

	Coefficienttest=True
	if Coefficienttest==True:
		Fehler=[]
		count=0
		for i in range(100):
			
			img1_gpu = clarray.to_device(queue, require(np.random.random(f_struct_gpu[0]), float32, 'F'))
			sino1_gpu = clarray.to_device(queue, require(np.random.random(f_struct_gpu[1]), float32, 'F'))
			img2_gpu = clarray.zeros(queue, f_struct_gpu[0], dtype=float32, order='F')
			sino2_gpu = clarray.zeros(queue, f_struct_gpu[1], dtype=float32, order='F')
			fanbeam_richy_gpu(sino2_gpu,img1_gpu,f_struct_gpu)
						
			fanbeam_richy_gpu_add(img2_gpu,sino1_gpu,f_struct_gpu)
			sino1=sino1_gpu.get().reshape(sino1_gpu.size)
			sino2=sino2_gpu.get().reshape(sino2_gpu.size)
			img1=img1_gpu.get().reshape(img1_gpu.size)
			img2=img2_gpu.get().reshape(img2_gpu.size)
			
			
			a=np.dot(img1,img2)
			b=np.dot(sino1,sino2)
			#import pdb; pdb.set_trace()
			if abs(a-b)/min(abs(a),abs(b))>0.001:
				print a,b
				count+=1
				Fehler.append((a,b))
		print 'Number of Errors: ',count,' Errors were ',Fehler
						
		
					
	from scipy import misc				
	Wallnut=misc.imread('Wallnut/Image.png')
	Wallnut=Wallnut/np.mean(Wallnut)
	Wallnut[np.where(Wallnut<=1.5)]=0
	Wallnut=scipy.misc.imresize(Wallnut,[328,328])
	
#	Wallnut=np.ones(Wallnut.shape)
	Wallnut_gpu=clarray.to_device(queue,require(Wallnut,float32,'F'))
	number_detectors=328
	Detectorwidth=114.8
	FOD=110
	FDD=300
	numberofangles=120
	geometry=[Detectorwidth,FDD,FOD,number_detectors]
	#misc.imshow(Wallnut)
	f_struct_gpu_wallnut = fanbeam_struct_richy_gpu(Wallnut.shape, numberofangles, Detectorwidth, FDD, FOD, number_detectors)
	sino2_gpu = clarray.zeros(queue, f_struct_gpu_wallnut[1], dtype=float32, order='F')
	
	fanbeam_richy_gpu(sino2_gpu,Wallnut_gpu,f_struct_gpu_wallnut)	
	#misc.imshow(sino2_gpu.get())
	Wallnut_gpu2=clarray.to_device(queue,require(Wallnut,float32,'F'))
	fanbeam_richy_gpu_add(Wallnut_gpu2,sino2_gpu,f_struct_gpu_wallnut)
	
	figure(1)
	imshow(Wallnut, cmap=cm.gray)
	title('original Wallnut')
	figure(2)
	imshow(sino2_gpu.get(), cmap=cm.gray)
	title('Fanbeam transformed image')
	figure(3)
	imshow(Wallnut_gpu2.get(), cmap=cm.gray)
	title('Backprojected image')
	show()
	#misc.imshow(Wallnut_gpu2.get())
	
	
	#U=Landweberiteration(sino2_gpu,f_struct_gpu_wallnut)
	#misc.imshow(U.get())
	
	
	
	sinonew=misc.imread('Wallnut/Sinogram.png')
	print 'asdf',np.max(sinonew)
	sinonew[np.where(sinonew<2000)]=0
	sinonew=np.array(sinonew,dtype=float)
	sinonew/=np.mean(sinonew)
	
	
	#h=scipy.hstack([sino2_gpu.get()/np.mean(sino2_gpu.get()),sinonew])
	#misc.imshow(h)
	#misc.imsave('Comparison_sinogram.png',h)
	Wallnut_gpu2new=clarray.to_device(queue,require(sinonew,float32,'F'))
	ULW=Landweberiteration(Wallnut_gpu2new,f_struct_gpu_wallnut,100)
	misc.imshow(ULW.get())
	#fanbeam_richy_gpu(Wallnut,U,f_struct_gpu_wallnut)
	#misc.imshow(Wallnut.get())
	#import pdb;pdb.set_trace()

	sinonew=[sinonew.T]
	import pdb; pdb.set_trace()
