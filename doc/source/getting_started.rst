Getting Started
****************

Basic principle of gratopy 
===============================

The cornerstone of the gratopy toolbox is formed by the :py:class:`gratopy.ProjectionSettings` class. When creating it one defines the geometry in which to work, and this class collects all relevant 
information to create the kernels and precomputes and saves
relevant quantities. Thus virtually all functions of gratopy require an object of this class. 
In particular, gratopy offers implementation for two different geometric settings, the parallel beam and the fanbeam setting. 

Typically an image is a function defined on a rectangular domain, but naturally for practical computations the values are saved in equi-distant quadratic pixels of size :math:`(N_x,N_y)`  (or :math:`(N_x,N_y,N_z)` for multiple slices) 
where the values associated are saved in a `pyopencl.Array  <https://documen.tician.de/pyopencl/array.html>`_ **img** and correspond to the average mass inside the pixel. We think of the object in question to be considered as a circular object contained in the corresponding image-rectangle.  
When using an image together with projectionsetting -- an instance of ProjectionSettings -- so the xy-shape has to coincide with the attribute **img_shape** of projectionsetting, we say they need to be **compatible**. The values dtype
of this array must be numpy.dtype(float32) or numpy.dtype(float), i.e. single or double precision, and can have either C or F contiguity. 
 

Similarly, a sinogram  is considered as :class:`pyopencl.Array`  **sino** of the form :math:`(N_s,N_a)` or :math:`(N_s,N_a,N_z)` for Ns the number of detecotors and Na the number of angles, where for the parallel beam setting the angular range :math`[0,\pi[` is considered. These dimensions must be **compatible** 
with the  projectionsetting of the class :class:`ProjectionSettings`  used together with, i.e., :math:`(N_s,N_a)` coincides with **sinogram_shape** attribute of projectionsetting. The width of the detector is given by detector_width, and the detector pixels are equidistantly partioning the detectorline with width 
:math:`\delta_s`. The angles on the other hand are given by the user and need not be equi-distant or even partion the entire angular domain. The values associated to pixel again correspond to the average
intensity values of a continuous sinogram counterpart.The values dtypeof this array must be numpy.dtype(float32) or numpy.dtype(float), i.e. single or double precision, and can have either C or F contiguity.
 
For the Radon transform the geometry is mainly determined by the ratio of image_width to detector_width, for most standard examples these parameters coincide. Note that a equivalent geometry in larger scale the values are scaled differently as well, but maintain there ratios.  

The considerations concerning **img** and **sino** remain valid for the fanbeam transform. However the geometry gets slightly more complex, as the distance from source to the center of rotation denoted
by RE and the distance from the source to the detector denoted by R. The angular range for the Fanbeam transform is given as a subset of :math:`[0,2\pi[`.

Standardly it is assumed that the given angles completely partition the angular range, which has implications for the backprojection. In case this is not desired, and rather a limited angle situation
is considered, see fullangle parameter of :py:class:`gratopy.ProjectionSettings`.

.. image:: grafics/Radon-1.png
    :width: 5000
    :alt: Depiction of parallel beam geometry
Geometry of the Radon transform.

	
.. image:: grafics/Fanbeam-1.png
	:alt: Depiction of fan beam geometry

Geometry of the Fanbeam transform.




First example: Radon transform
===============================

One can start in Python via
::
    #Initial import and definitions
    from numpy import *
    import pyopencl as cl
    import gratopy
    import matplotlib .pyplot as plt
    number_angles=60
    number_detector=300
    Nx=300

    #create pyopencl context
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
	
    #create phantom as testimage
    phantom=gratopy.phantom(queue,Nx)
	
    #create suitable ProjectionSettings
    PS=gratopy.ProjectionSettings(queue,gratopy.RADON,phantom.shape
        ,number_angles,number_detector)
		
    #Compute forward projection and backprojection of created sinogram	
    sino=gratopy.forwardprojection(phantom,PS)
    backproj=gratopy.backprojection(sino,PS)

    #Plot results
    plt.figure()
    plt.title("Generated Phantom")
    plt.imshow(phantom.get())
    
    plt.figure()
    plt.title("Sinogram")
    plt.imshow(sino.get())

    plt.figure()
    plt.title
    plt.imshow(backproj.get())
    plt.show()
	
	
Second example: Fanbeam transform
=================================
As a second example, we consider the fanbeam geometry, which has a detector that is 120 cm wide, the distance from the source to the center of rotation is 100 cm
while and the distance from source to detector are 200 cm. Via the :class:`show_geometry` method of the :class:`ProjectionSettings` to visualize the defined geometry.
::
    #Initial import and definitions
    from numpy import *
    import pyopencl as cl
    import gratopy
    import matplotlib .pyplot as plt
    number_angles=60
    number_detector=300
    image_shape=(500,500)
    Nx=300
	
    #create pyopencl context
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

	
    #Additional parameter
    my_detector_width=120
    my_R=200
    my_RE=100
	
    PS1=gratopy.ProjectionSettings(queue,gratopy.FANBEAM,img_shape=image_shape
        ,angles=number_angles,n_detectors=number_detector, 
        detector_width=my_detector_width,R=my_R,RE=my_RE)
        
    PS2=gratopy.ProjectionSettings(queue,gratopy.FANBEAM,img_shape=image_shape
        ,angles=number_angles,n_detectors=number_detector, 
        detector_width=my_detector_width,R=my_R,RE=my_RE,image_width=80)

    print("image_width chose by gratopy", PS1.image_width,
        "image_width when setting by hand",PS2.image_width)
   
    #Show associated geometry 
    fig,(axes1,axes2) =plt.subplots(1,2)

    PS1.show_geometry(pi/4,figure=fig,axes=axes1,show=False)
    PS2.show_geometry(pi/4,figure=fig,axes=axes2,show=False)
    axes1.set_title("Geometry chosen by gratopy")
    axes2.set_title("Geometry for hand-chosen image_width")
    plt.show()
    
Once the Geometry has been defined via the projectionsetting, forward and backprojections can be used just as for the Radon transform.
Note that the automatism of gratopy chooses image_width=57,46. When looking at the corresponding plot via show_geometry, creates this corresponds to capturing the entirety of an object inside 
the blue circle (with diameter 57,46), and thus the area represented by the image corresponds to the yellow rectangle and blue circle, in particular capturing any object contained in the blue circle. On the other hand, the outer red circle illustrates the diameter of an object wholy containing the image.


Further examples can be found in the test files in the `tests` folder inside gratopy, showing multiple examples and possible uses for the gratopy toolbox. 

.. image:: grafics/Figure_1.png
Plot from show_geometry for fanbeam setting with automatic and handchosen image_width.

