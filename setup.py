from setuptools import setup, find_packages
import dl_l8s2_uv

setup(name='dl_l8s2_uv',
      version=dl_l8s2_uv.__version__,
      description='Cloud masking of Landsat-8 and Sentinel-2 based on Deep Learning',
      author='Dan Lopez Puigdollers',
      author_email='dan.lopez@uv.es',
      packages=find_packages(exclude=["tests"]),
      package_data={'': ['*.hdf5']},
      include_package_data=True,
      install_requires=["numpy", "lxml", "spectral", "luigi", "h5py", "rasterio", "shapely"],
      zip_safe=False)