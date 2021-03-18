"""galsim_cosmos dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import galsim
# TODO(galsim_cosmos): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """This is a tf dataset wrapper for the COSMOS Training Sample 25.2 from GalSim."""

# TODO(galsim_cosmos): BibTeX citation
_CITATION = """{needed}"""

class GalSimCOSMOS(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for galsim_cosmos dataset."""

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {
      '0.0.1': 'Basic functionalities.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(galsim_cosmos): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Tensor(shape=[50,50],dtype=tf.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=("image","image"),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(galsim_cosmos): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://zenodo.org/record/3242143/files/COSMOS_25.2_training_sample.tar.gz')
    cat = galsim.COSMOSCatalog(dir= path / 'COSMOS_25.2_training_sample')
    # TODO(galsim_cosmos): Returns the Dict[split names, Iterator[Key, Example]]
    return {tfds.Split.TRAIN: self._generate_examples(cat)}

  def _generate_examples(self, data):
    """Yields examples."""
    # TODO(galsim_cosmos): Yields (key, example) tuples from the dataset
    for i in range(len(data)):
      gal = data.makeGalaxy(i)
      galxpsf = galsim.Convolve([gal,gal.original_psf])
      image = galxpsf.drawImage(nx=50,ny=50).array.astype("float32")

      yield '%d'%i, {'image': image}
