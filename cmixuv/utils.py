import numpy as np
import itertools
from numpy.lib.stride_tricks import as_strided


def mask_2D_to_3D(mascara, nchannels):
    return as_strided(mascara,
                      mascara.shape + (nchannels,),
                      mascara.strides + (0,))

def find_padding(v, divisor=8):
    v_divisible = max(divisor, int(divisor * np.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2


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