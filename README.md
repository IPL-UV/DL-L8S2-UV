# Benchmarking Deep Learning models for Cloud Masking in Landsat-8 and Sentinel-2

This repository contains source code used in

> [1] López-Puigdollers, D., Mateo-García, G., Gómez-Chova, L. “Benchmarking Deep Learning models for Cloud Masking in Landsat-8 and Sentinel-2” Submitted [pre-print](https://arxiv.org/abs/xxxx.xxxxx)

Code for masking clouds in Landsat-8 &amp; Sentinel-2.

## Requirements

The following code creates a new conda virtual environment with required dependencies.

```bash
conda create -n cmixuv -c conda-forge python=3.7 numpy scipy rasterio tensorflow=2 --y

conda activate pvl8

pip install spectral tqdm luigi 

```

## Inference Landsat-8 images

Expects an L1T Landsat-8 image from the [EarthExplorer](https://earthexplorer.usgs.gov/). 
The `--landsatimage` attribute points to the unzipped folder with a GeoTIF image for each band.

```
python inference.py CloudMaskL8 --l8image ./LC08_L1TP_002054_20160520_20170324_01_T1/ --namemodel rgbiswir
```
The folder `./LC08_L1TP_002054_20160520_20170324_01_T1` will contain a GeoTIF with the cloud mask.

## Inference Sentinel-2 images

Expects an L1C Sentinel-2 image from the [OpenHub](https://scihub.copernicus.eu/dhus). 
The `--s2image` attribute points to the unzipped `SAFE` folder. The `--resolution` attribute select the output resolution of the product (10, 20, 30 or 60)

```
python inference.py CloudMaskL8 --s2image ./S2A_MSIL1C_20160417T110652_N0201_R137_T29RPQ_20160417T111159.SAFE/ --namemodel rgbiswir --resolution 30
```
The folder `./LC08_L1TP_002054_20160520_20170324_01_T1` will contain a GeoTIF with the cloud mask.

 

