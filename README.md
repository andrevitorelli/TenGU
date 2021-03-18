# TenGU
Tensorflow-GalSim Universe

This is a collection of Tensorflow Datasets using that generate galaxy image stamps. For now, the following datasets are available:

inverse_cat: Creates 1e5 galaxy exponential stamps with a constant Moffat psf - with shear, size, flux, snr according to a "catalogue" in test.fits file in data. Modify the catalogue to what you want. Final size: ~1GiB

gal_gen: Creates 1e5 exponential galaxy random stamps with a constant Moffat psf. No download needed. Final size: ~1GiB

galsim_cosmos: Creates galaxy stamps from the COSMOS training set from GalSim ~87k galaxies - using the original psf. Modify the psf inside GalSimCOSMOS._generate_examples if you want. Final size: ~4.1GiB

galsim_HSC: Creates galaxy stamps from the HSC simulated images. Not implemented yet. Final size: >300GiB

All stamps are 50x50 (but can be changed before registering the tfds). 

## Installation:

Just enter each folder and tfds build it. The respective tfds will be available as tfds.load(name as string).
