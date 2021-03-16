"""GalGen dataset."""
from astropy.table import Table
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import galsim
from galaxies import exp_model, draw_gal_noise
_DESCRIPTION = "Toy Galaxies for simple proofs-of-concepts."
_CITATION = "{NEEDED}"
_URL = "https://github.com/andrevitorelli/TenGU/"


class GalGen(tfds.core.GeneratorBasedBuilder):
  """Simple Galaxy Image Generator for Tensorflow Operations."""

  VERSION = tfds.core.Version('0.0.0')
  RELEASE_NOTES = {'0.0.0': "Initial code."}

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description=_DESCRIPTION,
      homepage=_URL,
      features=tfds.features.FeaturesDict({
          'image': tfds.features.Tensor(shape=[50,50], dtype=tf.float32)
          }),
      supervised_keys=("image","image"),
   citation=_CITATION)

  def _split_generators(self,dl):
    """Returns generators according to split."""
    dl_path = dl.download(_URL + "data/test.fits" )
    data = Table.read(dl_path / "test.fits")
    intsplit = int(np.round(len(data)*0.7))
    train_slice = np.random.choice(np.arange(len(data)),intsplit,replace=False)
    test_slice = [i for i in np.arange(len(data))]
    _ = [test_slice.remove(train_selected) for train_selected in train_slice]
    return {'train': self._generate_examples(data[train_slice]),
             'test': self._generate_examples(data[test_slice]),
             }

  def _generate_examples(self, data):
    """Yields examples."""
    for i, galaxy in enumerate(data):
      image = draw_gal_noise(galaxy,None).astype("float32")
      g1, g2 = galaxy['g1'], galaxy['g2']

      yield '%d'%i, {'image': image}
