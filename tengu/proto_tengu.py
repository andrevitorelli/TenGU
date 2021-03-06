import galsim
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


_DESCRIPTION = """Tensorflow-GalSim Universe (TenGU)."""

_URL = 'https://github.com/andrevitorelli/TenGU'

_INPUTS = ['g1','g2','r0','snr']

_STAMP_SIZE=50

def DrawSimpleGalaxy(galaxy,psf=None,**kwargs):
    defaults = {
    'flux'         : 1,
    'method'       : "no_pixel",
    'stamp_size'   : _STAMP_SIZE,
    'scale'        : 0.2,
    'interpolator' : "linear"
    }
    defaults.update(kwargs)

    g1, g2, r0, snr = galaxy

    gal = galsim.Exponential(flux=1, scale_radius=r0)
    gal = gal.shear(g1=g1, g2=g2)
    if psf is not None:
        gal = galsim.Convolve([gal,psf])
    gal_image = gal.drawImage(nx=defaults['stamp_size'],
                              ny=defaults['stamp_size'], scale=.2, method=defaults['method'])
    noise = galsim.GaussianNoise()
    gal_image.addNoiseSNR(noise,snr)
    return gal_image.array

def DrawMoffatPSF(**kwargs):
    defaults = {
    'beta': 4.8,
    'radius' : 1,
    'interpolator': "linear"
    }
    defaults.update(kwargs)
    return galsim.Moffat(beta=defaults['beta'],half_light_radius=defaults['radius'])

def PSFs_from_img(img):
    return galsim.InterpolatedImage(img,x_interpolant=defaults['interpolator'])


class GalGen(tfds.core.GeneratorBasedBuilder):
    """Regression task aimed to predict the shear of galaxy images."""
    VERSION = tfds.core.Version('0.0.0')

    def _info(self):
        stamp_size = _STAMP_SIZE
        channels = 1
        return tfds.core.DatasetInfo(builder=self,
                                    description=_DESCRIPTION,
                                    features=tfds.features.FeaturesDict({
                                            'features':  tfds.features.Image(shape=(stamp_size,stamp_size,channels))}
                                            ),
                                    supervised_keys=None,
                                    homepage=_URL,
                                    citation=_CITATION
                                    )

    def _split_generators(self, cat,train_split=.7):
        """Returns SplitGenerators."""
        data = Table.read(cat)[_INPUTS]
        intsplit = int(np.round(len(data)*train_split))
        train_slice = np.random.choice(np.arange(len(data)),intsplit,replace=False)
        test_slice = [i for i in np.arange(len(data))]
        _ = [test_slice.remove(train_selected) for train_selected in train_slice]

        return {
        'train': self._generate_examples(data[train_slice]),
        'test': self._generate_examples(data[test_slice]),
        }

    def _generate_examples(self, data):
        """Yields examples."""

        for i, galaxy in enumerate(data):
            image = DrawSimpleGalaxy(galaxy)

            yield i, g1, g2, image
