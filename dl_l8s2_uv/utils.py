import rasterio
import itertools
import pkg_resources

import tensorflow as tf
import numpy as np

from numpy.lib.stride_tricks import as_strided
from dl_l8s2_uv import model

import os
import logging
import re


BANDS_MODEL = {"L8rgbiswir": [2, 3, 4, 5, 6, 7],  # 1-based band index
               "L8rgbi": [2, 3, 4, 5],  # 1-based band index
               "S2rgbi": [1, 2, 3, 8],  # 1-based band index
               "S2rgbiswir": [1, 2, 3, 8, 11, 12],  # 1-based band index
}

CLOUD_DETECTION_WEIGHTS = {
    "rgbiswir": pkg_resources.resource_filename('dl_l8s2_uv', 'weights/landsatbiomeRGBISWIR7.hdf5'),
    "rgbi": pkg_resources.resource_filename('dl_l8s2_uv', "weights/landsatbiomeRGBI6.hdf5")
}

# S2 name convention fields
NEW_FORMAT = "(S2\w{1})_MSIL\w{2}_(\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})_(\w{5})_(\w{4})_T(\w{5})_(\w{15})"
OLD_FORMAT = "(S2\w{1})_(\w{4})_(\w{3}_\w{6})_(\w{4})_(\d{8}T\d{6})_(\w{4})_V(\d{4}\d{2}\d{2}T\d{6})_(\d{4}\d{2}\d{2}T\d{6})"

# Set verbosity to error level
tf.get_logger().setLevel('ERROR')


####################################################################################
#                                    Tensorflow                                    #
####################################################################################
class Model:
    def __init__(self, satname, namemodel="rgbiswir", weights_path=None):
        self.satname = satname
        self.namemodel = namemodel
        self.bands_read = BANDS_MODEL[satname + namemodel]
        self.model_clouds = model.load_model(
            (None, None), weight_decay=0, bands_input=len(self.bands_read)
        )

        if weights_path is None:
            weights_path = CLOUD_DETECTION_WEIGHTS[namemodel]

        self.model_clouds.load_weights(weights_path)

    def predict(self, satobj):
        assert satobj.satname == self.satname, "{} image not compatible with {} model".format(satobj.satname,
                                                                                              self.satname)
        tf.keras.backend.clear_session()
        bands = satobj.load_bands(bands=self.bands_read)
        invalids = np.any(np.ma.getmaskarray(bands), axis=2)

        pad_r = find_padding(bands.shape[0])
        pad_c = find_padding(bands.shape[1])
        image_ti_batched = np.pad(
            bands, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), "reflect"
        )
        cloud_prob = predictbypatches(
            lambda patch: self.model_clouds.predict(patch[None])[0, ..., 0], image_ti_batched
        )
        slice_rows = slice(pad_r[0], None if pad_r[1] <= 0 else -pad_r[1])
        slice_cols = slice(pad_c[0], None if pad_c[1] <= 0 else -pad_c[1])
        cloud_prob = cloud_prob[(slice_rows, slice_cols)]

        invalids |= np.ma.getmaskarray(cloud_prob)

        # {0: invalid, 1: land, 2: cloud}
        cloud_prob_bin = (np.ma.filled(cloud_prob, 0) >= 0.5).astype(np.uint8) + 1
        cloud_prob_bin[invalids] = 0

        return cloud_prob_bin


def save_cloud_mask(satobj, cloud_prob_bin, out_path):
    crs = satobj.crs_proj()
    transform = satobj.rasterio_transform
    meta = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "nodata": 0,
        "width": cloud_prob_bin.shape[1],
        "height": cloud_prob_bin.shape[0],
        "count": 1,
        "compress": "lzw",
        "crs": crs,
        "blockxsize": 256,
        "blockysize": 256,
        "transform": transform,
    }
    # Save the cloud mask
    with rasterio.open(out_path, "w", **meta) as src_new:
        src_new.write_band(1, cloud_prob_bin)


def select_cuda_device(cuda_device="gpu", memory_growth=True):
    """
    Tweaks.
    Recommended Tensorflow >=2.1 with a CUDA GPU (if available)
    GPU is the default and preferred option, but it can incur memory issues:
    Force CPU                     - GPU can not handle a full image in any case or a CUDA GPU is not installed
    Force GPU with memory growth  - Allocating all memory at once may run out of memory so early

    :param cuda_device: 'gpu' or 'cpu'
    :param memory_growth: True or False
    """
    cuda_devices = {"CPU": "-1", "GPU": "0"}
    cuda_device = cuda_device.upper()

    assert cuda_device in list(cuda_devices.keys()), "{} is not a valid option. Choose gpu or cpu".format(cuda_device)

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices[cuda_device]

    if cuda_device == "GPU":
        # Allocate memory on demand
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) == 0:
            import logging
            logging.warning("GPU selected but not available. CPU will be used instead")
            return

        if memory_growth:
            tf.config.experimental.set_memory_growth(physical_devices[0], memory_growth)


def predictbypatches(pred_function, img_HWC, patchsize=1024, buffersize=16):

    shape = img_HWC.shape[:2]
    mask_save_invalid = np.zeros(shape, dtype=np.bool)
    output_save = np.zeros(shape, dtype=img_HWC.dtype)

    # predict in tiles of tile_size x tile_size pixels
    for i, j in itertools.product(range(0, shape[0], patchsize), range(0, shape[1], patchsize)):
        slice_current = (
            slice(i, min(i + patchsize, shape[0])),
            slice(j, min(j + patchsize, shape[1])),
        )
        slice_pad = (
            slice(max(i - buffersize, 0), min(i + patchsize + buffersize, shape[0])),
            slice(max(j - buffersize, 0), min(j + patchsize + buffersize, shape[1])),
        )

        slice_save_i = slice(
            slice_current[0].start - slice_pad[0].start,
            None
            if (slice_current[0].stop - slice_pad[0].stop) == 0
            else slice_current[0].stop - slice_pad[0].stop,
        )
        slice_save_j = slice(
            slice_current[1].start - slice_pad[1].start,
            None
            if (slice_current[1].stop - slice_pad[1].stop) == 0
            else slice_current[1].stop - slice_pad[1].stop,
        )

        # slice_save is normally slice(buffersize,-buffersize),slice(buffersize,-buffersize) except in the borders
        slice_save = (slice_save_i, slice_save_j)

        img_slice_pad = img_HWC[slice_pad + (slice(None),)]

        maskcarainvalid = np.any(np.ma.getmaskarray(img_slice_pad), axis=-1, keepdims=False)

        mascarainvalidcurrent = maskcarainvalid[slice_save]

        # call predict only if there are pixels not invalid
        if not np.all(mascarainvalidcurrent):
            mask_save_invalid[slice_current] = mascarainvalidcurrent
            vals_to_predict = np.ma.filled(img_slice_pad, 0)

            pred_continuous_tf = pred_function(vals_to_predict)[slice_save]
            output_save[slice_current] = pred_continuous_tf

    return np.ma.masked_array(output_save, mask_save_invalid)


def find_padding(v, divisor=8):
    v_divisible = max(divisor, int(divisor * np.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2


####################################################################################
#                                    Landsat-8                                     #
####################################################################################
def mask_2D_to_3D(mascara, nchannels):
    return as_strided(mascara,
                      mascara.shape + (nchannels,),
                      mascara.strides + (0,))


####################################################################################
#                                 Sentinel-2 utils                                 #
####################################################################################
def s2_name_split(s2l1c):
    """
    https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention

    ```
    s2l1c = "S2A_MSIL1C_20151218T182802_N0201_R127_T11SPD_20151218T182756.SAFE"
    mission, sensing_date_str, pdgs, relorbitnum, tile_number_field, product_discriminator = s2_name_split(s2l1c)
    ```

    S2A_MSIL1C_20151218T182802_N0201_R127_T11SPD_20151218T182756.SAFE
    MMM_MSIXXX_YYYYMMDDTHHMMSS_Nxxyy_ROOO_Txxxxx_<Product Discriminator>.SAFE
    MMM: is the mission ID(S2A/S2B)
    MSIXXX: MSIL1C denotes the Level-1C product level/ MSIL2A denotes the Level-2A product level
    YYYYMMDDHHMMSS: the datatake sensing start time
    Nxxyy: the PDGS Processing Baseline number (e.g. N0204)
    ROOO: Relative Orbit number (R001 - R143)
    Txxxxx: Tile Number field
    SAFE: Product Format (Standard Archive Format for Europe)

    :param s2l1c:
    :return:
    """
    basename = os.path.basename(os.path.splitext(s2l1c)[0])
    matches = re.match(NEW_FORMAT, basename)
    if matches is not None:
        return matches.groups()


def s2_old_format_name_split(s2l1c):
    """
    https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention

    ```
    s2l1c = "S2A_OPER_PRD_MSIL1C_PDMC_20151206T093912_R090_V20151206T043239_20151206T043239.SAFE"
    mission, opertortest, filetype, sitecenter,  creation_date_str, relorbitnum, sensing_time_start, sensing_time_stop = s2_old_format_name_split(s2l1c)
    ```

    :param s2l1c:
    :return:
    """
    basename = os.path.basename(os.path.splitext(s2l1c)[0])
    matches = re.match(OLD_FORMAT, basename)
    if matches is not None:
        return matches.groups()


def getKeyByValue(dictOfElements, valueToFind):
    # Get the key from dictionary which has the given value
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            mykey = item[0]
            break

    # Assuming that values are unique
    return mykey


def list_dir(path_pv, list_):
    for entry in os.scandir(path_pv):
        if entry.is_dir():
            list_ = list_dir(entry.path, list_)
        elif entry.is_file:
            list_.append(entry.path)

    return list_


def search_string_list(i_list, string):
    entry = i_list[np.argwhere([string in elem for elem in i_list]).squeeze()]
    if entry.size <= 1:
        entry = entry.tolist()
        if len(entry) == 0:
            entry = None
    elif len(entry) > 1:
        entry = np.array(entry)

    return entry


def round_mask(gt, masked=True, invalid_value=-1):
    if isinstance(gt, np.ma.MaskedArray):
        gt = np.ma.filled(gt, invalid_value)

    gt_round = np.round(gt, 0)  # round to the closest integer
    if masked:
        gt_round = np.ma.MaskedArray(data=gt_round, mask=gt_round == invalid_value)

    return gt_round


def round_cords(cords_array, mode="exact",warn=True):
    """
    Given 2 coordinates as an array in format row col (e.g. np.array([[r0,c0],[r1,c1]]) returns the rounded values of
    the rows and cols according to the criteria given by mode.

    :param cords_array:
    :param mode:
    :param warn:
    :return:
    """
    assert any((m == mode for m in ['inner', 'outer', 'round', 'exact'])), "unexpected slice_mode"

    cords_array = np.array(cords_array)
    cords_array_round = cords_array.round()

    diff = np.abs(cords_array - cords_array_round)
    if np.max(diff) > 1e-6:
        if mode == 'round':
            if warn:
                logging.info("The slice in the new transform is not exact: {} using: {}".format(cords_array.T,
                                                                                            cords_array_round.T))
        elif mode == 'inner':
            cords_array_round[0, :] = np.where(diff[0, :] > 1e-6,
                                                           np.ceil(cords_array[0, :]).astype(np.int64),
                                                           cords_array_round[0, :])
            cords_array_round[1, :] = np.where(diff[1, :] > 1e-6,
                                                           np.floor(cords_array[1, :]).astype(np.int64),
                                                           cords_array_round[1, :])
            if warn:
                logging.info("The slice in the new transform is not exact: {} using: {}".format(cords_array.T,
                                                                                            cords_array_round.T))
        elif mode == 'outer':
            cords_array_round[0, :] = np.where(diff[0, :] > 1e-6,
                                                           np.floor(cords_array[0, :]).astype(np.int64),
                                                           cords_array_round[0, :])
            cords_array_round[1, :] = np.where(diff[1, :] > 1e-6,
                                                           np.ceil(cords_array[1, :]).astype(np.int64),
                                                           cords_array_round[1, :])
            if warn:
                logging.info(
                "The slice in the new transform is not exact: {} using: {}".format(cords_array.T,
                                                                                   cords_array_round.T))
        else:
            raise ValueError(
                "The mode for the slice is exact but the slice in the new transform is not exact: {}".format(
                    cords_array.T))

    return cords_array_round.astype(np.int64)


def round_slice(slice_, mode="exact",warn=True):
    """
    Receives an slice with possibly float values and returns the slice rounded and casted to int

    :param slice_:
    :param mode:
    :param warn:
    :return:
    """
    coords_round = round_cords(np.array([[slice_[0].start, slice_[1].start], [slice_[0].stop, slice_[1].stop]]),
                               mode=mode, warn=warn)
    return (slice(coords_round[0, 0], coords_round[1, 0]), slice(coords_round[0, 1], coords_round[1, 1]))


def assert_valid_slice(slice_, shape=None):
    """
    :param slice_:
    :param shape:
    :return:
    """
    if shape is None:
        shape = (9999999, 9999999)
    for i, s in enumerate(slice_):
        assert s.start is not None and (s.start >= 0) and (s.start <= shape[i]), \
            "{} (0 row 1 col) start point is {} for an array with shape {}".format(i, s, shape)
        assert s.stop is not None and (s.stop >= 0) and (s.stop <= shape[i]), \
            "{} (0 row 1 col) stop point is {} for an array with shape {}".format(i, s, shape)


def shape_slice(slice_):
    """
    It assumes slice_old is a normalized slice
    """
    return tuple([s.stop - s.start for s in slice_])
