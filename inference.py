import os
import luigi

from dl_l8s2_uv import utils
from dl_l8s2_uv.satreaders import l8image, s2image


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
            self.model_clouds = utils.Model(satname=self.satname(), namemodel=self.namemodel)

        return self.model_clouds

    def output(self):
        path_img = os.path.join(self.satobj().folder, "dluvclouds_" + self.namemodel + ".tif")
        return luigi.LocalTarget(path_img)

    def run(self):
        satobj = self.satobj()
        model = self.cloud_detection_model()
        cloud_prob_bin = model.predict(satobj)

        # Save the cloud mask
        utils.save_cloud_mask(satobj, cloud_prob_bin, self.output().path)


class CloudMaskL8(CloudMask):
    landsatimage = luigi.Parameter(description="Dir where the landsat image is stored (unzipped)")
    namemodel = luigi.ChoiceParameter(description="name to save the binary cloud mask",
                                      choices=["rgbi", "rgbiswir", "allnt"])

    def satobj(self):
        if not hasattr(self, "satobj_computed"):
            setattr(self, "satobj_computed", l8image.L8Image(self.landsatimage))
        return getattr(self, "satobj_computed")

    def satname(self):
        return "L8"


class CloudMaskS2(CloudMask):
    s2image = luigi.Parameter(description="Dir where the landsat image is stored (unzipped)")
    resolution = luigi.ChoiceParameter(choices=["10", "20", "30", "60"], default="30",
                                       description="Spatial resolution to get the cloud mask")
    namemodel = luigi.ChoiceParameter(description="name to save the binary cloud mask",
                                      choices=["rgbi", "rgbiswir", "allnt"])

    def satobj(self):
        if not hasattr(self, "satobj_computed"):
            setattr(self, "satobj_computed", s2image.S2Image(self.s2image, size_def=self.resolution))
        return getattr(self, "satobj_computed")

    def satname(self):
        return "S2"


if __name__ == "__main__":
    luigi.run(local_scheduler=True)





