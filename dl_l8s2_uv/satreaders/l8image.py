"""
Classes and functions for reading L8 images and manually annotated cloud masks from the Biome and 38-Cloud
cloud cover dataset.

https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data

"""
import os
from datetime import datetime
from datetime import timezone

import numpy as np
import rasterio
import dl_l8s2_uv.utils as utils


def read_metadata(metadata_file):
    assert os.path.exists(metadata_file), "metadata file %s does not exist" % metadata_file

    with open(metadata_file, "r") as mf:
        lineas = mf.readlines()

    dictio_sal = dict()
    for l in lineas:
        dato = [d.strip() for d in l.strip().split("=")]
        if dato[0] == "GROUP":
            dictio_sal[dato[1]] = dict()
            curr_dict = dictio_sal[dato[1]]
        elif (dato[0] == "END_GROUP") or (dato[0] == "END"):
            continue
        else:
            curr_dict[dato[0]] = dato[1].replace('"', "")

    return dictio_sal


def compute_toa_single(img, band, metadata, sun_elevation_correction=True):
    """
    Readiomatric correction implemented in:
    https://www.usgs.gov/land-resources/nli/landsat/using-usgs-landsat-level-1-data-product

    :param img:
    :param band:
    :param metadata:
    :param sun_elevation_correction: whether or not to do the sun elevation correction
    :return:
    """
    band_name = str(band)

    if band < 10:
        dictio_rescaling = metadata["RADIOMETRIC_RESCALING"]
        mult_key = "REFLECTANCE_MULT_BAND_" + band_name
        img = img*float(dictio_rescaling[mult_key])

        add_key = "REFLECTANCE_ADD_BAND_" + band_name
        img += float(dictio_rescaling[add_key])

        if sun_elevation_correction:
            dictio_rescaling = metadata["IMAGE_ATTRIBUTES"]
            sun_elevation_angle_key = "SUN_ELEVATION"
            img /= np.sin(float(dictio_rescaling[sun_elevation_angle_key]) / 180. * np.pi)

    else:
        #  (band == 10) or (band == 11)
        dictio_rescaling = metadata["RADIOMETRIC_RESCALING"]
        mult_key = "RADIANCE_MULT_BAND_" + band_name
        img = img * float(dictio_rescaling[mult_key])

        add_key = "RADIANCE_ADD_BAND_" + band_name
        img += float(dictio_rescaling[add_key])

        dictio_rescaling = metadata["TIRS_THERMAL_CONSTANTS"]
        k1_key = "K1_CONSTANT_BAND_" + band_name
        img = np.log(float(dictio_rescaling[k1_key])/img +1)
        k2_key = "K1_CONSTANT_BAND_" + band_name
        img = float(dictio_rescaling[k2_key])/img

    return img


def load_l8_clouds_fmask(l8bqa):
    """
    https://www.usgs.gov/land-resources/nli/landsat/landsat-collection-1-level-1-quality-assessment-band
    https://www.usgs.gov/land-resources/nli/landsat/cfmask-algorithm

    :param l8bqa:
    :return:
    """
    return (l8bqa & (1 << 4)) != 0


SAMPLE_BANDS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]  # All bands except B8 (pancromatic with different spatial resolution)
ORIENTATION = {"NORTH_UP": "north", "SOUTH_UP": "south"}


class L8Image:
    """
    Class to load L1T Landsat-8 image

    :param folder_tiffs: folder where the tiffs and metadata stored.
    :param slice_rows_cols: list of slices=[slice(100,200),slice(100,200)]
    to read only specific locations of the image
    """
    def __init__(self, folder_tiffs, slice_rows_cols=None):
        if folder_tiffs.endswith('/'):
            folder_tiffs = folder_tiffs[:-1]
        self.folder_tiffs = folder_tiffs
        self.folder = folder_tiffs
        self.name = os.path.basename(folder_tiffs)
        self.satname = "L8"
        self.metadata = self._read_metadata()

        dictio_metadata = self.metadata["PRODUCT_METADATA"]

        self.polygon = np.array([[float(dictio_metadata["CORNER_LL_LON_PRODUCT"]),
                               float(dictio_metadata["CORNER_LL_LAT_PRODUCT"])],
                              [float(dictio_metadata["CORNER_UL_LON_PRODUCT"]),
                               float(dictio_metadata["CORNER_UL_LAT_PRODUCT"])],
                              [float(dictio_metadata["CORNER_UR_LON_PRODUCT"]),
                               float(dictio_metadata["CORNER_UR_LAT_PRODUCT"])],
                              [float(dictio_metadata["CORNER_LR_LON_PRODUCT"]),
                               float(dictio_metadata["CORNER_LR_LAT_PRODUCT"])]])

        proj_param = self.metadata["PROJECTION_PARAMETERS"]
        zone = "" if "UTM_ZONE" not in proj_param else " +zone=%s" % proj_param["UTM_ZONE"]
        self.crs_string_biome = "+proj=%s%s +ellps=%s +datum=%s +units=m +%s" % (proj_param["MAP_PROJECTION"].lower(),
                                                                            zone,
                                                                            proj_param["ELLIPSOID"],
                                                                            proj_param["DATUM"],
                                                                            ORIENTATION[proj_param["ORIENTATION"]])

        pm = self.metadata["PRODUCT_METADATA"]
        trans = [float(pm["CORNER_UL_PROJECTION_X_PRODUCT"]), float(pm["CORNER_UL_PROJECTION_Y_PRODUCT"])]
        self.rasterio_transform = self.src_rasterio().transform
        self.transform_numpy = np.array([[30, 0, trans[0]], [0, -30, trans[1]]])

        self.nrows = int(pm["REFLECTIVE_LINES"])
        self.ncols = int(pm["REFLECTIVE_SAMPLES"])

        self.start_date = datetime.strptime(dictio_metadata["DATE_ACQUIRED"] + " " +
                                            dictio_metadata["SCENE_CENTER_TIME"][:8],
                                            "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        self.end_date = self.start_date

        if slice_rows_cols is None:
            self.slice = (slice(0, self.nrows), slice(0, self.ncols))
        else:
            self.slice = tuple(slice_rows_cols)

    def crs_proj(self):
        import rasterio
        fileband_name = os.path.join(self.folder_tiffs, self.name + "_B2.TIF")
        with rasterio.open(fileband_name, "r") as src:
            src_crs = src.crs
        return src_crs

    @property
    def transform(self):
        import rasterio
        fileband_name = os.path.join(self.folder_tiffs, self.name + "_B2.TIF")
        with rasterio.open(fileband_name, "r") as src:
            tr = src.transform

        return np.array([[tr.a, tr.b, tr.c], [tr.d, tr.e, tr.f]])

    def __str__(self):
        return self.folder_tiffs

    def _read_metadata(self):
        metadata_file = os.path.join(self.folder_tiffs, self.name + "_MTL.txt")
        return read_metadata(metadata_file)

    def load_mask(self, slice_=None):
        bqa = self.load_bqa(slice_=slice_)

        return bqa == 1

    def load_bqa(self, slice_=None):
        if slice_ is None:
            slice_ = (slice(None), slice(None))

        with rasterio.open(os.path.join(self.folder_tiffs, self.name + "_BQA.TIF"), "r") as src:
            bqa = src.read(1, window=slice_)
        return bqa[slice_]

    def load_fmask(self, slice_=None):
        bqa = self.load_bqa(slice_=slice_)
        return load_l8_clouds_fmask(bqa)

    def load_band(self, band, compute_toa_flag=True,
                  sun_elevation_correction=True,
                  slice_=None):
        """
        https://www.usgs.gov/land-resources/nli/landsat/using-usgs-landsat-level-1-data-product

        Lλ = MLQcal + AL
        Lλ          = TOA spectral radiance (Watts/( m2 * srad * μm))
        ML         = Band-specific multiplicative rescaling factor from the metadata (RADIANCE_MULT_BAND_x, where x is the band number)
        AL          = Band-specific additive rescaling factor from the metadata (RADIANCE_ADD_BAND_x, where x is the band number)
        Qcal        = Quantized and calibrated standard product pixel values (DN)

        :param band:
        :param compute_toa_flag:
        :param sun_elevation_correction:
        :param slice_: slice to read
        :return:
        """
        band_name = str(band)
        fileband_name = os.path.join(self.folder_tiffs, self.name + \
                                     "_B" + band_name + ".TIF")
        if slice_ is None:
            slice_ = self.slice

        with rasterio.open(fileband_name, "r") as src:
            img = src.read(1, window=slice_).astype(np.float32)

        if compute_toa_flag:
            return compute_toa_single(img, band, self.metadata,
                                      sun_elevation_correction=sun_elevation_correction)
        return img

    def src_rasterio(self):
        fileband_name = os.path.join(self.folder_tiffs, self.name + \
                                     "_B2.TIF")
        return rasterio.open(fileband_name, "r")

    def load_bands(self, bands=None, masked=True,
                   compute_toa_flag=True,
                   sun_elevation_correction=True,
                   axis_stack=2,
                   slice_=None):
        """
        load the bands `bands` and stack them over the `axis_stack` axis.

        :param bands: Bands to read (band 8 has different spatial resolution)
        :param masked: if the bands should be read with the mask
        :param compute_toa_flag:
        :param axis_stack: axis to stack the bands (0 or 2)
        :param sun_elevation_correction: wether or not to apply the sun elevation correction
        :param slice_:

        :return: the 3D multispectral image
        """
        img = None
        if bands is None:
            bands = SAMPLE_BANDS

        assert len(bands) == 1 or (8 not in bands), "Cannot load panchromatic band (B8) together with other bands because it has different resolution"
        assert axis_stack in {0, 2}, "Expected to stack on first or last dimension"
        for i,k in enumerate(bands):
            b = self.load_band(k,compute_toa_flag=compute_toa_flag,
                               sun_elevation_correction=sun_elevation_correction,
                               slice_=slice_)
            if img is None:
                if axis_stack == 2:
                    img = np.ndarray(b.shape+(len(bands),),dtype=b.dtype)
                else:
                    img = np.ndarray((len(bands),) + b.shape, dtype=b.dtype)

            if axis_stack == 2:
                img[..., i] = b
            else:
                img[i] = b

        if masked:
            mask = self.load_mask(slice_)
            mask = utils.mask_2D_to_3D(mask, img.shape[axis_stack])
            img = np.ma.masked_array(img, mask)

        return img


class Biome(L8Image):
    """
    Class to deal with the L8 cloud validation dataset downloaded from https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data

    :param folder_tiffs: folder where the tiffs, metadata and envi fixedmask are stored.
    """
    def __init__(self, s2_folder):
        L8Image.__init__(self, s2_folder)

        self.hdr_file = os.path.join(self.folder_tiffs, self.name + "_fixedmask.hdr")
        assert os.path.exists(self.hdr_file), "fixedmask not found: {}".format(self.hdr_file)

    def _load_fixemask(self, slice_=None):
        import spectral.io.envi as envi

        if slice_ is None:
            slice_ = (slice(None), slice(None))

        return envi.open(self.hdr_file)[slice_]

    def load_mask(self, slice_=None):
        mask_envi = self._load_fixemask(slice_)
        return mask_envi[..., 0] == 0

    def load_clouds(self, slice_=None):
        """
        The interpretation for the bits in each manual mask is as follows:
        Value  Interpretation
        ----------------------
        0	   Fill
        64	   Cloud Shadow
        128	   Clear
        192	   Thin Cloud
        255	   Cloud

        :return:
        """
        mask_envi = self._load_fixemask(slice_)
        return cloudmask(mask_envi[..., 0])


def cloudmask(fixedmask):
    """
    The interpretation for the bits in each manual mask is as follows:
    Value  Interpretation
    ----------------------
    0	   Fill
    64	   Cloud Shadow
    128	   Clear
    192	   Thin Cloud
    255	   Cloud

    :return:
    """

    mask = (fixedmask == 0)
    mask_envi = np.uint8((fixedmask == 192) | (fixedmask == 255))

    return np.ma.masked_array(mask_envi, mask)


def cloud_shadow_mask(fixedmask):
    """
    The interpretation for the bits in each manual mask is as follows:
    Value  Interpretation
    ----------------------
    0	   Fill
    64	   Cloud Shadow
    128	   Clear
    192	   Thin Cloud
    255	   Cloud

    :return:
    """

    mask = (fixedmask == 0)

    mask_envi = np.zeros(fixedmask.shape,dtype=np.uint8)
    mask_envi[(fixedmask == 192) | (fixedmask == 255)] = 2
    mask_envi[(fixedmask == 64)] = 1

    return np.ma.masked_array(mask_envi, mask)


class L8_38Clouds(L8Image):
    """
    Class to deal with the 38-Clouds cloud validation dataset.

    GT masks downloaded from https://www.kaggle.com/sorour/38cloud-cloud-segmentation-in-satellite-images
    (Entire_scene_gts folder)
    Landsat-8 images downloaded from the Earth Explorers' portal: https://earthexplorer.usgs.gov/

    :param folder_tiffs: folder where the tiffs, metadata and edited_corrected_gts.
    """
    def __init__(self, folder_tiffs):
        L8Image.__init__(self, folder_tiffs)

        self.gt_file = os.path.join(self.folder_tiffs, 'edited_corrected_gts' + self.name + ".TIF")
        assert os.path.exists(self.hdf5_file), "edited_corrected_gts not found: {}".format(self.gt_file)

    def _load_fixemask(self, slice_=None):
        if slice_ is None:
            slice_ = (slice(None), slice(None))

        with rasterio.open(self.gt_file) as src_gt:
            img = src_gt.read(1, window=slice_)

        return img

    def load_mask(self, slice_=None):
        img = self._load_fixemask(slice_)

        return img == 0

    def load_clouds(self, slice_=None):
        """
        The interpretation for the bits in each manual mask is as follows:
        Value  Interpretation
        ----------------------
        0	   Fill
        1	   Clear
        2	   Cloud

        :return:
        """

        img = self._load_fixemask(slice_)
        mask = (img == 0)
        mask_clouds = np.uint8(img == 2)
        return np.ma.masked_array(mask_clouds, mask)
