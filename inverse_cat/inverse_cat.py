"""InverseCat dataset."""
from astropy.table import Table
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import galsim
from galaxies import draw_gal_noise
_DESCRIPTION = "This tfds uses a simple fits table to create galaxy stamps."
_CITATION = "{NEEDED}"
_URL = "https://github.com/andrevitorelli/TenGU/"


class InverseCat(tfds.core.GeneratorBasedBuilder):
  """Simple Galaxy Image Generator from Catalogue for Tensorflow Operations."""

  VERSION = tfds.core.Version('0.0.0')
  RELEASE_NOTES = {'0.0.0': "Initial code."}

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description=_DESCRIPTION,
      homepage=_URL,
      features=tfds.features.FeaturesDict({
          'image': tfds.features.Tensor(shape=[50,50,1], dtype=tf.float32),
          'label': tfds.features.Tensor(shape=[2], dtype=tf.float32)
          }),
      supervised_keys=("image","label"),
   citation=_CITATION)

  def _split_generators(self,dl):
    """Returns generators according to split."""
    dl_path = dl.download(_URL+"raw/main/data/test.fits" )
    data = Table.read(str(dl_path))

    return {tfds.Split.TRAIN: self._generate_examples(data)}

  def _generate_examples(self, data):
    """Yields examples."""
    for i, galaxy in enumerate(data):
      image = draw_gal_noise(galaxy,None).astype("float32")
      image.shape = (50,50,1)
      label = np.array([galaxy['g1'], galaxy['g2']],dtype="float32" )

      yield '%d'%i, {'image': image,
                     'label': label }
