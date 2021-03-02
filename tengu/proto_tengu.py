import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from astropy.table import table
from astropy.io import fits
import galsim

def DrawGal(stamp_size = 50,
            gal_r0 = 1.0,            # arcsec
            g1=0,g2=0,
            snr=200,
            psf = None):
    gal = galsim.Exponential(flux=1, scale_radius=gal_r0)
    gal = gal.shear(g1=g1, g2=g2)
    if psf is not None:
        gal = galsim.Convolve([gal,psf])
    gal_image = gal.drawImage(nx=stamp_size,ny=stamp_size, scale=pixel_scale, method='no_pixel')
    noise = galsim.GaussianNoise()
    gal_image.addNoiseSNR(noise,snr)
    return gal_image

class InverseCat:
    
    def __init__(self,cat,**kwargs):
        self.defaults =  {'stamp_size' : 50,
                    'gal_flux' : 1,
                    'interpolator' : 'linear',
            
        }    
        self.defaults.update(kwargs)
        
        self.table = Table.read(cat)
        self.location = cat
        self.size = len(self.table)
        self.psfs = [None for i in range(self.size)]
        return print("Catalog loaded")
       
    def create_psfs(self,psfs=None,**kwargs):
        default =  {'beta' : 4.8,
                    'radius' : 1,
        }
        default.update(kwargs)
        if psfs == None:
            self.psfs = [galsim.Moffat(beta=default['beta'],half_light_radius=default['radius']) for i in range(self.size)]
        else:
            self.psfs = [galsim.InterpolatedImage(psf,x_interpolant=self.defaults['interpolator']) for psf in psfs]
        return print("PSFs created")
            
            
    def _DrawGalaxy(self,galaxy,psf=None):
        g1, g2, r0, snr = galaxy['g1','g2','r0','snr']
        gal_img=DrawGal(stamp_size = self.defaults['stamp_size'],
            gal_r0 = r0,            # arcsec
            g1=g1, g2=g2,
            snr=snr,
            psf = psf)
        return gal_img
       
    def draw_galaxies(self):
        gal_images = [self._DrawGalaxy(galaxy,psf).array for galaxy,psf in zip(self.table,self.psfs)]
        self.table['images'] = gal_images
        return print('Galaxy images generated.')
        
    def write_images(self,location):
        for n, image in enumerate(self.table['images']):
            fits.writeto(location+f'gal_img_{n}.fits',image,overwrite=True)  
        return print(f'{self.size} images written.')
