import galsim

def exp_model(galaxy):
  gal = galsim.Exponential(flux=galaxy['flux'] , half_light_radius=galaxy['re'])
  return gal

def moffat_psf(g1,g2,**kwargs):
  defaults = {
    'beta'  : 4.765,
    'fwhm'  : 1}
  defaults.update(kwargs)
  psf = galsim.Moffat(beta=defaults['beta'], half_light_radius=defaults['fwhm'])
  return psf

def draw_gal_noise(galaxy,
                   psf = moffat_psf(0.01,-0.02),
                   model=exp_model,**kwargs):
  defaults = {
  'pixel_scale'  : 0.2,
  'stamp_size'   : 50,
  'method'       : "no_pixel",
  'interpolator' : "linear"
  }
  defaults.update(kwargs)

  gal = model(galaxy)
  gal = gal.shear(g1=galaxy['g1'],g2=galaxy['g2'])

  if psf is not None:
    gal = galsim.Convolve([gal,psf])

  gal_image = gal.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])
  noise = galsim.GaussianNoise()
  gal_image.addNoiseSNR(noise,galaxy['snr'])

  return gal_image.array
