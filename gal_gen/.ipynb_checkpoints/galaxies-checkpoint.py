import galsim

def moffat_psf(g1,g2,**kwargs):
  defaults = {
    'beta' : 4.765,
    'fwhm' : 1}
  defaults.update(kwargs)
  psf = galsim.Moffat(beta=defaults['beta'], half_light_radius=defaults['fwhm'])
  return psf

def generate_galaxy(psf = moffat_psf(0.01,-0.02),**kwargs):
  "random galaxy generator"

  defaults = {'re_mean' : 3.0,
              're_scatter' : 0.1,
              'g_range' : [-.6,-6],
              'g_rms' : 0.3,
              'snr_mean' : 100,
              'snr_scatter' : 30,
              'pixel_scale' : 0.2,
              'stamp_size' : 50,
              'method' : "no_pixel",
              'interpolator' : "linear"}

  defaults.update(kwargs)

  g1 = truncnorm.rvs(-.6, .6, loc=0, scale=0.2)
  g2 = truncnorm.rvs(-.6, .6, loc=0, scale=0.2)
  re = truncnorm.rvs(.5, 5, loc=3, scale=0.2)
  gal = galsim.Exponential(flux=defaults['flux'] ,                                                half_light_radius=re)
  gal = gal.shear(g1=galaxy['g1'],g2=galaxy['g2'])

  if psf is not None:
    gal = galsim.Convolve([gal,psf])

  gal_image = gal.drawImage(nx=defaults['stamp_size'],
                            ny=defaults['stamp_size'],
                            scale=defaults['pixel_scale'],
                            method=defaults['method'])
  noise = galsim.GaussianNoise()
  gal_image.addNoiseSNR(noise,galaxy['snr'])

  return g1, g2, gal_image.array
