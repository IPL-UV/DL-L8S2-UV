import luigi

from cmixuv import utils, model
from cmixuv.satreaders import l8image
import os

import numpy as np
import rasterio


CLOUD_DETECTION_WEIGHTS = {
    "rgbiswir": "/media/disk/databases/CMIX/setupgonzalo/experiments/landsatbiomeRGBISWIR7.hdf5", # TODO create checkpoints folder with model weights
    "rgbi": "/media/disk/databases/CMIX/setupgonzalo/experiments/landsatbiomeRGBI6.hdf5",
}

BANDS_MODEL = {"L8rgbiswir": [2, 3, 4, 5, 6, 7], # 1-based band index
               "L8rgbi": [2, 3, 4, 5], # 1-based band index
               "S2rgbi" : [2, 3, 4, 8], # 1-based band index
               "S2rgbiswir": [2, 3, 4, 8, 11, 12], # 1-based band index
}


class CloudMask(luigi.Task):
    namemodel = luigi.ChoiceParameter(description="name to save the binary cloud mask",
                                      choices=["rgbi", "rgbiswir"], default="rgbiswir")

    def satobj(self):
        raise NotImplementedError("Must add a satname")

    def satname(self):
        raise NotImplementedError("Must add a satname")

    def cloud_detection_model(self):
        if hasattr(self, "model_clouds"):
            return self.model_clouds
        else:
            self.model_clouds = model.load_model(
                (None, None), weight_decay=0, bands_input=len(BANDS_MODEL)
            )
            self.model_clouds.load_weights(CLOUD_DETECTION_WEIGHTS[self.namemodel])
        return self.model_clouds

    def output(self):
        path_img = os.path.join(self.landsatimage,"cmixuvclouds_" + self.namemodel + ".tif")
        return luigi.LocalTarget(path_img)

    def run(self):
        satobj = self.satobj()
        model = self.cloud_detection_model()
        bands_read = BANDS_MODEL[self.satname()+self.namemodel]

        bands = satobj.load_bands(bands=bands_read)
        invalids = np.any(np.ma.getmaskarray(bands), axis=2)

        pad_r = utils.find_padding(bands.shape[0])
        pad_c = utils.find_padding(bands.shape[1])
        image_ti_batched = np.pad(
            bands, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), "reflect"
        )
        cloud_prob = utils.predictbypatches(
            lambda patch: model.predict(patch[None])[0, ..., 0], image_ti_batched
        )
        slice_rows = slice(pad_r[0], None if pad_r[1] <= 0 else -pad_r[1])
        slice_cols = slice(pad_c[0], None if pad_c[1] <= 0 else -pad_c[1])
        cloud_prob = cloud_prob[(slice_rows, slice_cols)]

        invalids |= np.ma.getmaskarray(cloud_prob)

        cloud_prob_bin = (np.ma.filled(cloud_prob, 0) >= 0.5).astype(
            np.uint8
        ) + 1  # {0: invalid, 1: land, 2: cloud}
        cloud_prob_bin[invalids] = 0
        src_rasterio = satobj.src_rasterio()
        meta = {
            "driver": "GTiff",
            "dtype": rasterio.uint8,
            "nodata": 0,
            "width": cloud_prob_bin.shape[1],
            "height": cloud_prob_bin.shape[0],
            "count": 1,
            "compress": "lzw",
            "crs": src_rasterio.crs,
            "blockxsize": 256,
            "blockysize": 256,
            "transform": src_rasterio.transform,
        }

        # Save the cloud mask
        luigitarget = self.output()
        with rasterio.open(luigitarget.path, "w", **meta) as src_new:
            src_new.write_band(1, cloud_prob_bin)


class CloudMaskL8(luigi.Task):
    landsatimage = luigi.Parameter(description="Dir where the landsat image is stored (unzipped)")
    namemodel = luigi.ChoiceParameter(description="name to save the binary cloud mask",
                                      choices=["rgbi", "rgbiswir", "allnt"])

    def satobj(self):
        if not hasattr(self, "satobj_computed"):
            setattr(self, "satobj_computed", l8image.L8Image(self.landsatimage))
        return getattr(self, "satobj_computed")

    def satname(self):
        return "L8"

class CloudMaskS2(luigi.Task):
    s2image = luigi.Parameter(description="Dir where the landsat image is stored (unzipped)")
    resolution = luigi.ChoiceParameter(choices=["10", "20", "30", "60"], default="30",
                                       description="Spatial resolution to get the cloud mask")
    namemodel = luigi.ChoiceParameter(description="name to save the binary cloud mask",
                                      choices=["rgbi", "rgbiswir", "allnt"])

    def satobj(self):
        if not hasattr(self, "satobj_computed"):
            setattr(self, "satobj_computed", s2image.S2Image(self.s2image))
        return getattr(self, "satobj_computed")

    def satname(self):
        return "S2"


if __name__ == "__main__":
    luigi.run(local_scheduler=True)







