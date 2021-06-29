Getting Started
****************

Basic principle of gratopy 
===============================

ProjectionSettings
-------------------

The cornerstone of the gratopy toolbox is formed by the :py:class:`gratopy.ProjectionSettings` class, which defines the considered geometry , and collects all relevant 
information to create the kernels and precomputes and saves
relevant quantities. Thus, virtually all functions of gratopy require an object of this class. 
In particular, gratopy offers implementation for two different geometric settings, the parallel beam and the fanbeam setting. 

The geometry of the parallel beam setting is mainly defined by the **image_width** -- the physical diameter of the object in question -- and **detector_width** -- the physical width of the detector --,
both parameters of a projectionsetting. For most standard examples of Radon transform these parameters coincide, i.e., the detector is exactly as long as the diameter of object and thus captures exactly all rays passing through the object. 
For the fanbeam setting additionally the distance from source to the center of rotation denoted by **RE** and the distance from the source to the detector denoted by **R** are necessary to define 
the geometry, see figures below. Note that these transforms are scaling-invariant, meaning rescaling the geometry creates operators which are also rescaled versions of the original ones.

The angular range for the parallel beam setting is :math:`[0,\pi[`, while for fanbeam it is :math:`[0,2\pi[`. 
Standardly it is assumed that the given angles completely partition the angular range, which has implications for the backprojection. In case this is not desired, and rather a limited angle situation
is considered, see fullangle parameter of :py:class:`gratopy.ProjectionSettings`.



.. image:: grafics/Radon-1.png
    :width: 5000
    :alt: Depiction of parallel beam geometry
Geometry of the parallel beam setting.

	
.. image:: grafics/Fanbeam-1.png
    :width: 5000
    :alt: Depiction of fan beam geometry
Geometry of the fanbeam setting.


The main functions of gratopy are  :class:`gratopy.forwardprojection` and :class:`gratopy.backprojection`, which use a projectionsetting as a basis for computation and allow to project 
an image **img** onto an sinogram **sino** and backproject **sino** onto **img**. Next we describe, which form such images and sinograms need to have and how to interpret corresponding values.

 
Images in gratopy
-------------------

An image of the investigated object is represented in gratopy by a `pyopencl.Array  <https://documen.tician.de/pyopencl/array.html>`_ **img** of dimensions :math:`(N_x,N_y)`   
-- or :math:`(N_x,N_y,N_z)` for multiple slices -- representing a rectangular grid of equi-distant quadratic pixels,
where the associated values correspond to the average mass inside the pixel. We think of the investigated object as being a circular and contained in the corresponding image-rectangle whose larger sidelength correspond to the **image_width** attribute of the projectionsetting.  
When using an image together with projectionsetting -- an instance of :class:`ProjectionSettings` --  :math:`(Nx,Ny)` has to coincide with the attribute **img_shape** of projectionsetting, we say they need to be **compatible**. The  dtype
of this array must be numpy.dtype(float32) or numpy.dtype(float), i.e., single or double precision, and can have either C or F contiguity. 
 
Sinograms in gratopy
------------------------

Similarly, a sinogram  is represented by a :class:`pyopencl.Array`  **sino** of the form :math:`(N_s,N_a)` or :math:`(N_s,N_a,N_z)` for :math:`N_s` the number of detectors and :math:`N_a` the number of angles from which projections are considered. These dimensions must be **compatible** 
with the  projectionsetting of the class :class:`ProjectionSettings`  used together with, i.e., :math:`(N_s,N_a)` coincides with the  **sinogram_shape** attribute of projectionsetting. 
The width of the detector is given by the attribute **detector_width** of projectionsetting, and the detector pixels are equidistantly partioning the detectorline with width 
:math:`\delta_s`. The angles on the other hand need not be equi-distant or even partion the entire angular range. The values associated to pixels in the sinogram again correspond to the average
intensity values of a continuous sinogram counterpart. The dtype of this array must be numpy.dtype(float32) or numpy.dtype(float), i.e., single or double precision, and can have either C or F contiguity.
 






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
    plt.imshow(phantom.get(),cmap="gray")
    
    plt.figure()
    plt.title("Sinogram")
    plt.imshow(sino.get(),cmap="gray")

    plt.figure()
    plt.title("Backprojection")
    plt.imshow(backproj.get(),cmap="gray")
    plt.show()

The following depcits the plots created by the example 1.

.. image:: grafics/Phantom.png
    :width: 5000

.. image:: grafics/Sinogram.png
    :width: 5000
.. image:: grafics/Backprojection.png
    :width: 5000


Second example: Fanbeam transform
=================================
As a second example, we consider the fanbeam geometry, which has a detector that is 120 cm wide, the distance from the source to the center of rotation is 100 cm
while and the distance from source to detector are 200 cm. We do not choose the image_width but rather let gratopy automatically choose a suitable image_width. Via the :class:`gratopy.show_geometry` method of the :class:`gratopy.ProjectionSettings` we visualize the defined geometry.
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
	
    #Fanbeam setting with automatic image_width
    PS1=gratopy.ProjectionSettings(queue,gratopy.FANBEAM,
        img_shape=image_shape
        ,angles=number_angles,n_detectors=number_detector, 
        detector_width=my_detector_width,R=my_R,RE=my_RE)
    
    print("image_width chose by gratopy", PS1.image_width)

    #Fanbeam setting with set image_width
    my_image_width=80    
    PS2=gratopy.ProjectionSettings(queue,gratopy.FANBEAM,
        img_shape=image_shape
        ,angles=number_angles,n_detectors=number_detector, 
        detector_width=my_detector_width,R=my_R,RE=my_RE,
        image_width=my_image_width)

   
    #Plot geometries associated to these projectionsettings
    fig,(axes1,axes2) =plt.subplots(1,2)
    PS1.show_geometry(pi/4,figure=fig,axes=axes1,show=False)
    PS2.show_geometry(pi/4,figure=fig,axes=axes2,show=False)
    axes1.set_title("Geometry chosen by gratopy as "
        +str(PS1.image_width))
    axes2.set_title("Geometry for hand-chosen image_width as "
        +str(my_image_width))
    plt.show()
    
Once the geometry has been defined via the projectionsetting, forward and backprojections can be used just as for the Radon transform in example 1.
Note that the automatism of gratopy chooses image_width=57.46 cm. When looking at the corresponding plot via :class:`show_geometry`, the image_width is such that the entirety of an object inside 
the blue circle (with diameter 57.46) is exactly captured by each projection, and thus the area represented by the image corresponds to the yellow rectangle and blue circle, the smallest rectangle to capture the entire object. On the other hand, the outer red circle illustrates the diameter of an object wholy containing the image.


Further examples can be found in the example-files in the *tests* folder inside gratopy, showing multiple examples and possible uses for the gratopy toolbox. 

.. image:: grafics/Figure_1.png
    :width: 5000
    :align: center
Plot from show_geometry for fanbeam setting with automatic and handchosen image_width.

