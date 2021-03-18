"""gal_gen dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import galsim
from galaxies import generate_galaxy
_DESCRIPTION = "Tensorflow-GalSim Universe: Toy Galaxies for simple proofs-of-concepts."
_CITATION = "{NEEDED}"
_URL = "https://github.com/andrevitorelli/TenGU/"


class GalGen(tfds.core.GeneratorBasedBuilder):
  """Random galaxy image generator."""

  VERSION = tfds.core.Version('0.0.0')
  RELEASE_NOTES = {'0.0.0': "Initial code."}

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # Description and homepage used for documentation
      description=_DESCRIPTION,
      homepage=_URL,
      features=tfds.features.FeaturesDict({
          'image': tfds.features.Tensor(shape=[50,50], dtype=tf.float32),
          'label': tfds.features.Tensor(shape=[2], dtype=tf.float32)
          }),
      supervised_keys=("image","label"),
   citation=_CITATION)

  def _split_generators(self,dl):
    """Returns generators according to split."""
    return {tfds.Split.TRAIN: self._generate_examples(data)}

  def _generate_examples(self, data):
    """Yields examples."""
    for i, in range(100000):
      galaxy, image = generate_galaxy().astype("float32")
      label = np.array([galaxy['g1'], galaxy['g2']],dtype="float32" )

      yield '%d'%i, {'image': image,
                     'label': label }
