from setuptools import setup

setup(name='cmixuv',
      version=0.1,
      description='Cloud masking of Landsat-8 and Sentinel-2 based on Deep Learning',
      author='Dan Lopez Puigdollers',
      author_email='dan.lopez@uv.es',
      include_package_data=True,
      install_requires=["numpy", "lxml", "rasterio", "shapely", "spectral", "tensorflow", "luigi", "matplotlib"],
      zip_safe=False)