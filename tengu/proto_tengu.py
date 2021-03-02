import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from astropy.table import table
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

    def write_images(self,location):
        for n, image in enumerate(self.table['images']):
            fits.writeto(location+f'gal_img_{n}.fits',image)


class galgen(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for galgen dataset."""

    VERSION = tfds.core.Version('0.0.0')
    RELEASE_NOTES = {
      '0.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(256, 256, 3)),
                'label': tfds.features.ClassLabel(names=['no', 'yes']),
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits."""

        extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
        # dl_manager returns pathlib-like objects with `path.read_text()`,
        # `path.iterdir()`,...
        return {
            'train': self._generate_examples(path=extracted_path / 'train_images'),
            'test': self._generate_examples(path=extracted_path / 'test_images'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[Key, Example]]:
        """Generator of examples for each split."""
        for img_path in path.glob('*.jpeg'):
            # Yields (key, example)
            yield img_path.name, {
              'image': img_path,
              'label': 'yes' if img_path.name.startswith('yes_') else 'no',
            }
