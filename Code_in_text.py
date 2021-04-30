from numpy import *
from matplotlib.pyplot import *
import pyopencl as cl
import grato 
import matplotlib.image as mpimg


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


img=mpimg.imread('Shepp_Logan.png')[:,:,0:3]
angles=180
Ns=500

if True:
	PS=grato.projection_settings(queue,"parallel",img.shape,angles,Ns)
	img_gpu = cl.array.to_device(queue, require(img, "float32", 'F'))
	sino_gpu=grato.forwardprojection(None,img_gpu,PS)
	backproj_gpu=grato.backprojection(None,sino_gpu,PS)

	figure(1)
	imshow(img_gpu.get())
	figure(2)
	imshow(sino_gpu.get()/np.max(sino_gpu.get()),cmap=cm.gray)
	figure(3)
	imshow(backproj_gpu.get()/np.max(backproj_gpu.get()),cmap=cm.gray)
	#show()

	matplotlib.pyplot.imsave("images/parallel/img.png",img_gpu.get(),cmap=cm.gray)
	matplotlib.pyplot.imsave("images/parallel/sino.png",sino_gpu.get(),cmap=cm.gray)
	matplotlib.pyplot.imsave("images/parallel/backproj.png",backproj_gpu.get()/np.max(backproj_gpu.get()),cmap=cm.gray)

if True:
	
	
	image_width=6.
	angles=np.linspace(np.pi*1/3,np.pi*2/3,180)
	for detector_shift in [0,1,-1]:
		for detector_width in [3,6]:
	
			PS=grato.projection_settings(queue,"parallel",img.shape,angles,Ns,image_width=image_width,detector_width=detector_width,detector_shift=detector_shift,fullangle=False)
			img_gpu = cl.array.to_device(queue, require(img, "float32", 'F'))
			sino_gpu=grato.forwardprojection(None,img_gpu,PS)
			backproj_gpu=grato.backprojection(None,sino_gpu,PS)



			figure(4)
			imshow(img_gpu.get())
			figure(5)
			imshow(sino_gpu.get()/np.max(sino_gpu.get()),cmap=cm.gray)
			figure(6)
			imshow(backproj_gpu.get()/np.max(backproj_gpu.get()),cmap=cm.gray)
		#	show()

			matplotlib.pyplot.imsave("images/parallel/img_smalldetector_shift"+str(detector_shift)+"width_"+str(detector_width)+".png",img_gpu.get()/np.max(img_gpu.get()),cmap=cm.gray)
			matplotlib.pyplot.imsave("images/parallel/sino_smalldetectorshift"+str(detector_shift)+"width_"+str(detector_width)+".png",sino_gpu.get()/np.max(sino_gpu.get()),cmap=cm.gray)
			matplotlib.pyplot.imsave("images/parallel/backproj_smalldetectorshift"+str(detector_shift)+"width_"+str(detector_width)+".png",backproj_gpu.get()/np.max(backproj_gpu.get()),cmap=cm.gray)


	detector_shift=-1
	detector_width=6
	PS=grato.projection_settings(queue,"parallel",img.shape,angles,Ns,image_width=image_width,detector_width=detector_width,detector_shift=detector_shift,fullangle=True)

	img_gpu = cl.array.to_device(queue, require(img, "float32", 'F'))
	sino_gpu=grato.forwardprojection(None,img_gpu,PS)
	backproj_gpu=grato.backprojection(None,sino_gpu,PS)

	matplotlib.pyplot.imsave("images/parallel/False_backprojection"+str(detector_shift)+"width_"+str(detector_width)+".png",backproj_gpu.get()/np.max(backproj_gpu.get()),cmap=cm.gray)

	

if True:
	Wallnut=mpimg.imread('Wallnut/ground_truth.png')
	Wallnut=Wallnut/np.mean(Wallnut); Wallnut[np.where(Wallnut<=1.5)]=0
	dtype=np.float;	number_detectors=328;	Detectorwidth=114.8
	FOD=110;	FDD=300;	numberofangles=120
	
	PS = grato.projection_settings(queue,"fan",img_shape=Wallnut.shape  
				,angles = numberofangles,detector_width=Detectorwidth, R=FDD 
				,RE=FOD, n_detectors=number_detectors,data_type=dtype)

	Wallnut_gpu=cl.array.to_device(queue,require(Wallnut,dtype,'F'))
	Wallnut_sino=grato.forwardprojection(None,Wallnut_gpu,PS)	

	Wallnut_backprojection=grato.backprojection(None,Wallnut_sino,PS)

	figure(7)
	imshow(Wallnut, cmap=cm.gray)
	title('original Wallnut')
	figure(8)
	imshow(Wallnut_sino.get(), cmap=cm.gray)
	title('Fanbeam transformed image')
	figure(9)
	imshow(Wallnut_backprojection.get(), cmap=cm.gray)
	title('Backprojected image')
	
	
	matplotlib.pyplot.imsave("images/Wallnut/img.png",Wallnut,cmap=cm.gray)
	matplotlib.pyplot.imsave("images/Wallnut/sino.png",Wallnut_sino.get(),cmap=cm.gray)
	matplotlib.pyplot.imsave("images/Wallnut/backproj.png",Wallnut_backprojection.get(),cmap=cm.gray)


	#Wallnut_sino=mpimg.imread('Wallnut/Sinogram.png')
	#Wallnut_sino_gpu=cl.array.to_device(queue,require(Wallnut_sino,dtype,'F'))
	#ULW=grato.Landweberiteration(Wallnut_sino_gpu,PS,20)
	#imshow(ULW.get(),cmap=cm.gray)
	#title("Landweber reconstruction")
	#show()

	#matplotlib.pyplot.imsave("images/Wallnut/Landweber.png",ULW.get())

	

if True: ##CG convergence
	A=np.array([[2,1,3,8],[0,5,7,6],[1,4,3,2],[1,2,3,4]])
	y=np.array([3,2,1,4])
	x=np.array([0,0,0,0])
	d=y-np.dot(A,x)
	p=np.dot(A.T,d)
	q=d*0
	snew=p+0.
	for k in range(0,10):
		residual=np.linalg.norm(snew)**0.5
		print(k,residual)
		if  residual<=0:
			break
		sold=snew+0.
		
		q=np.dot(A,p)
		
		alpha=np.linalg.norm(sold)/np.linalg.norm(q)
		alpha=alpha**2
		cl.enqueue_barrier(queue)
		x=x+alpha*p
		d=d-alpha*q
#		print("alpha",alpha)
#		print(np.linalg.norm(sold.get())**2/np.linalg.norm(q.get())**2)
#		import pdb;pdb.set_trace()
		cl.enqueue_barrier(queue)
		snew=np.dot(A.T,d)
		beta=np.linalg.norm(snew)/np.linalg.norm(sold)
		beta=beta**2
		p=snew+beta*p
#		print("beta",beta)

	

if True:
	
	dtype=float32
	number_detectors=328
	Detectorwidth=114.8
	FOD=110
	FDD=300
	numberofangles=120
	
	Nx=500
	
	x0=cl.array.zeros(queue,(Nx,Nx),dtype=dtype,order="F")

	
	
	Wallnut_sino=mpimg.imread('Wallnut/Sinogram.png')
	Wallnut_sino_gpu=cl.array.to_device(queue,require(Wallnut_sino,dtype,'F'))
	
	PS = grato.projection_settings(queue,"fan",img_shape=(Nx,Nx), angles = numberofangles,detector_width=Detectorwidth, R=FDD, RE=FOD, n_detectors=number_detectors,data_type=dtype)
	ULW=grato.Landweberiteration(Wallnut_sino_gpu,PS,50)
	UCG=grato.CG_iteration(Wallnut_sino_gpu,PS,epsilon=0.0,x0=x0,number_iterations=10,relaunch=True)
	figure(10)
	imshow(ULW.get(),cmap=cm.gray)
	figure(11)
	imshow(UCG.get(),cmap=cm.gray)
	
	figure(12)
	imshow(np.hstack([ULW.get(),UCG.get()]))
	show()

	matplotlib.pyplot.imsave("images/Wallnut/Landweber.png",ULW.get(),cmap=cm.gray)
	matplotlib.pyplot.imsave("images/Wallnut/CG.png",UCG.get(),cmap=cm.gray)

	
show()
