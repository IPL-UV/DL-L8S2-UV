# Benchmarking Deep Learning models for Cloud Detection in Landsat-8 and Sentinel-2 images

This repository contains source code used in

> [1] López-Puigdollers, D., Mateo-García, G., Gómez-Chova, L. “Benchmarking Deep Learning models for Cloud Detection in Landsat-8 and Sentinel-2 images” [paper](https://www.mdpi.com/2072-4292/13/5/992/htm)

![GA abstract](figs/GA_MDPI_RS_Benchmark.png)
<!---
![NN architecture](figs/neural_network.png) 
--->

## Requirements

The following code creates a new conda virtual environment with required dependencies.

```bash
conda create -n dl_l8s2_uv -c conda-forge python=3.7 tensorflow=2 matplotlib --y

conda activate dl_l8s2_uv

python setup.py install

```

## Inference Landsat-8 images

Expects an L1T Landsat-8 image from the [EarthExplorer](https://earthexplorer.usgs.gov/).
The `--l8image` attribute points to the unzipped folder with a GeoTIFF image for each band.

```
python inference.py CloudMaskL8 --l8image ./LC08_L1TP_002054_20160520_20170324_01_T1/ --namemodel rgbiswir
```
The folder `./LC08_L1TP_002054_20160520_20170324_01_T1` will contain a GeoTIFF with the cloud mask.

## Inference Sentinel-2 images

Expects an L1C Sentinel-2 image from the [OpenHub](https://scihub.copernicus.eu/dhus).
The `--s2image` attribute points to the unzipped `SAFE` folder. The `--resolution` attribute select the output resolution of the product (10, 20, 30 or 60)

```
python inference.py CloudMaskS2 --s2image ./S2A_MSIL1C_20160417T110652_N0201_R137_T29RPQ_20160417T111159.SAFE/ --namemodel rgbiswir --resolution 30
```
The folder `./S2A_MSIL1C_20160417T110652_N0201_R137_T29RPQ_20160417T111159.SAFE` will contain a GeoTIFF with the cloud mask.

## Inference Notebook

We have also included a notebook that uses the model and plots the results inline [here](./notebooks/Example%20-%20Cloud%20masking%20with%20DL%20in%20L-8%20and%20S-2%20images.ipynb).

## Cite

If you use this code please cite:

```
@article{lopez-puigdollers_benchmarking_2021,
  title={Benchmarking Deep Learning Models for Cloud Detection in Landsat-8 and Sentinel-2 Images},
  author={L{\'o}pez-Puigdollers, Dan and Mateo-Garc{\'\i}a, Gonzalo and G{\'o}mez-Chova, Luis},
  journal={Remote Sensing},
  doi={10.3390/rs13050992},
  link={https://www.mdpi.com/2072-4292/13/5/992/htm},
  volume={13},
  number={5},
  pages={992},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

## Related work

* [Multitemporal cloud masking in the Google Earth Engine](https://github.com/IPL-UV/ee_ipl_uv)
* [Landsat-8 to Proba-V transfer learning and Domain adaptation for cloud detection](https://github.com/IPL-UV/pvl8dagans)

## Acknowledgements

This work has been developed in the context of the projects TEC2016-77741-R and PID2019-109026RB-I00 (MINECO-ERDF) granted to Luis Gómez-Chova.
