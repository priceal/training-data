 

Simulation
		Measured intensity = object X PSF + background + shot noise
		Objects: points, circles, spheres, rods, blobs, stars, polygons, etc…
		PSF --- Airy disk or more sophisticated? Include different types of microscopy?

	
we need a basic library of particle shape "ground truths"
    how to project 3D particle (sphere) into 2D circle?
    
    
particle shapes:
    
ellipses
Gaussian shapes


routines to create particle shape functions:
    
    1. return a function which is a multi-Gaussian
        takes as argument: list of particle centers, widths, and amplitudes
                            l