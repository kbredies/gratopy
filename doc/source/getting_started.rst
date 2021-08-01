Getting started
===============

Basic principles of gratopy
---------------------------

We start by explaining some recurring relevant quantities and concepts in gratopy, in particular the :class:`ProjectionSettings <gratopy.ProjectionSettings>` class as well as the use of images and sinograms in the context of gratopy.

ProjectionSettings
''''''''''''''''''

The cornerstone of the gratopy toolbox is formed by the :py:class:`gratopy.ProjectionSettings` class, which defines the considered geometry, collects all relevant 
information to create the OpenCL kernels and precomputes as well as saves
relevant quantities. Thus, virtually all functions of gratopy require an object of this class, usually referred to as **projectionsetting**. 
In particular, gratopy offers the implementation for two different geometric settings, the **parallel beam** and the **fanbeam** setting. 

The geometry of the parallel beam setting is mainly defined by the **image_width** -- the physical diameter of the object in question in arbitrary units, e.g., 3 corresponding to 3cm (or m etc.) -- and **detector_width** -- the physical width of the detector in arbitrary units --,
both parameters of a **projectionsetting**. For most of standard examples for the Radon transform, these parameters coincide, i.e., the detector is exactly as wide as the diameter of the imaged object and thus captures exactly all rays passing through the object. 

For the fanbeam setting, the physical distance from source to the center of rotation, denoted by **RE**, and the physical distance from the source to the detector, denoted by **R**, are additionally necessary to define 
the geometry, see the figures below. 

Moreover, the projection requires discretization parameters, i.e., the shape of the image to project from and the number of detector pixels to map to. Note that these transforms are scaling-invariant in the sense that
rescaling all *physical* quantities by the same factor creates operators which are rescaled versions of the original ones. On the other hand, changing the number of pixels leaves the 
physical system invariant and simply reflects a finer/coarser discretization.

The angular range for the parallel beam setting is :math:`[0,\pi[`, while for the fanbeam setting, it is :math:`[0,2\pi[`. 
By default, it is assumed that the given angles completely partition the angular range. In case this is not desired  and a limited-angle situation
is considered, the **fullangle** parameter of :py:class:`gratopy.ProjectionSettings` can be adapted, impacting for instance the backprojection operator.
Note also that the projections considered are rotation-invariant in the sense, that projection of a rotated image yields a sinogram which is translated in the angular dimension.


.. image:: graphics/radon-1.png
    :width: 5000
    :alt: Depiction of parallel beam geometry
    
Geometry of the parallel beam setting.

	
.. image:: graphics/fanbeam-1.png
    :width: 5000
    :alt: Depiction of fan beam geometry
    
Geometry of the fanbeam setting.


The main functions of gratopy are  :func:`forwardprojection <gratopy.forwardprojection>` and :func:`backprojection <gratopy.backprojection>`, which use a **projectionsetting** as the basis for computation and allow to project 
an image **img** onto an sinogram **sino** and to backproject **sino** onto **img**, respectively. Next, we describe the requirements for such images and sinograms, and how to interpret their corresponding values.

 
Images in gratopy
'''''''''''''''''

An image **img** is represented in gratopy by a :class:`pyopencl.array.Array` of dimensions :math:`(N_x,N_y)`
-- or :math:`(N_x,N_y,N_z)` for multiple slices -- representing a rectangular grid of equi-distant quadratic pixels of size :math:`\delta_x=\mathrm{image_width}/\max\{N_x,N_y\}`,
where the associated values correspond to the average mass inside the area covered by each pixel. Usually, we think of the investigated object as being circular and contained in
the rectangular image domain of **img**. More generally, **image_width** corresponds to the larger side-length of an rectangular :math:`(N_x,N_y)` grid  of quadratic image pixels
which allow to consider *slim* objects.  
In any case the object should be contained in the rectangular image-domain (with sides parallel to the x and y axes), in particular slim in vertical or horizontal direction in case of non-square images.  **(???)**.  
When using an image together with **projectionsetting** -- an instance of :class:`gratopy.ProjectionSettings` --  the values :math:`(N_x,N_y)` have to coincide with the attribute **img_shape** of **projectionsetting**, we say they need to be **compatible**. The data type
of this array must be :attr:`numpy.float32` or :attr:`numpy.float64`, i.e., single or double precision, and can have either *C* or *F* `contiguity <https://documen.tician.de/pyopencl/array.html#pyopencl.array.Array>`_. 
 
Sinograms in gratopy
''''''''''''''''''''

Similarly, a sinogram  **sino** is represented by a :class:`pyopencl.array.Array`  of the shape :math:`(N_s,N_a)` or :math:`(N_s,N_a,N_z)` for :math:`N_s` being the number of detectors and :math:`N_a` being the number of angles for which projections are considered. 
When used together with a **projectionsetting** of class :class:`gratopy.ProjectionSettings`, these dimensions must be **compatible**, i.e., :math:`(N_s,N_a)` has to coincide with the  **sinogram_shape** attribute of **projectionsetting**. 
The width of the detector is given by the attribute **detector_width** of **projectionsetting** and the detector pixels are equi-distantly partitioning the detector line with detector pixel width 
:math:`\delta_s`. The angles, on the other hand, do not need to be equi-distant or even partition the entire angular range. The values associated with pixels in the sinogram again correspond to the average
intensity values of a continuous sinogram counterpart. The data type of this array must be :attr:`numpy.float32` or :attr:`numpy.float64`, i.e., single or double precision, and can have either *C* or *F* `contiguity`_.
 


First example: Radon transform
------------------------------

One can start in Python via
::

    # initial import
    from numpy import *
    import pyopencl as cl
    import gratopy
    import matplotlib.pyplot as plt
    
    # discretization parameters
    number_angles=60
    number_detector=300
    Nx=300

    # create pyopencl context
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
	
    # create phantom as test image (a pyopencl.array.Array of dimensions (Nx,Nx))
    phantom=gratopy.phantom(queue,Nx)
	
    # create suitable projectionsettings
    PS=gratopy.ProjectionSettings(queue, gratopy.RADON, phantom.shape,
                                  number_angles, number_detector)
		
    # compute forward projection and backprojection of created sinogram
    # results are pyopencl arrays	
    sino = gratopy.forwardprojection(phantom, PS)
    backproj = gratopy.backprojection(sino, PS)

    # plot results
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

The following depicts the plots created by this example.

.. image:: graphics/phantom-1.png
    :width: 5000

.. image:: graphics/sinogram-1.png
    :width: 5000
    
.. image:: graphics/backprojection-1.png
    :width: 5000


Second example: Fanbeam transform
---------------------------------

As a second example, we consider a fanbeam geometry which has a detector that is 120 (cm) wide, the distance from the source to the center of rotation is 100 (cm),
while the distance from source to detector are 200 (cm). We do not choose the **image_width** but rather let gratopy automatically determine a suitable **image_width**. We visualize the defined geometry via the :class:`gratopy.ProjectionSettings.show_geometry` method.
::

    # initial import
    from numpy import *
    import pyopencl as cl
    import gratopy
    import matplotlib .pyplot as plt
    
    # discretization parameters
    number_angles=60
    number_detector=300
    image_shape=(500,500)
	
    # create pyopencl context
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # physical parameters
    my_detector_width=120
    my_R=200
    my_RE=100
	
    # fanbeam setting with automatic image_width
    PS1 = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
                        img_shape=image_shape, angles=number_angles,
			n_detectors=number_detector, 
                        detector_width=my_detector_width, R=my_R,
			RE=my_RE)
    
    print("image_width chosen by gratopy: {:.2f}".format((PS1.image_width)))

    # fanbeam setting with set image_width
    my_image_width=80    
    PS2 = gratopy.ProjectionSettings(queue, gratopy.FANBEAM,
        img_shape=image_shape,
        angles=number_angles, n_detectors=number_detector, 
        detector_width=my_detector_width, R=my_R, RE=my_RE,
        image_width=my_image_width)

    # plot geometries associated to these projectionsettings
    fig, (axes1, axes2) = plt.subplots(1,2)
    PS1.show_geometry(pi/4, figure=fig, axes=axes1, show=False)
    PS2.show_geometry(pi/4, figure=fig, axes=axes2, show=False)
    axes1.set_title("Geometry chosen by gratopy as: {:.2f}".format((PS1.image_width)))
    axes2.set_title("Geometry for manually-chosen image_width as: {:.2f}".format((my_image_width)))
    plt.show()
    
Once the geometry has been defined via the **projectionsetting**, forward and backprojections can be used just as for the Radon transform in the first example.
Note that the automatism of gratopy chooses **image_width** =57.46 (cm). When looking at the corresponding plot via :class:`gratopy.ProjectionSettings.show_geometry`, the **image_width** is such that the entirety of an object inside 
the blue circle (with diameter 57.46) is exactly captured by each projection, and thus, the area represented by the image corresponds to the yellow rectangle and blue circle which is the smallest rectangle to capture the entire object. On the other hand, the outer red circle illustrates the diameter of the largest object entirely containing the image.

.. image:: graphics/figure-1.png
    :width: 5000
    :align: center

Plot produced by :class:`gratopy.ProjectionSettings.show_geometry` for the fanbeam setting with automatic and manually chosen **image_width**.

Further examples can be found in the source files of the `test examples <test_examples.html>`_. 

