import itertools

import numpy as np
import tensorflow as tf


def conv_blocks(
    ip_,
    nfilters,
    axis_batch_norm,
    reg,
    name,
    batch_norm,
    remove_bias_if_batch_norm=False,
    dilation_rate=(1, 1),
):
    use_bias = not (remove_bias_if_batch_norm and batch_norm)

    conv = tf.keras.layers.SeparableConv2D(
        nfilters,
        (3, 3),
        padding="same",
        name=name + "_conv_1",
        depthwise_regularizer=reg,
        pointwise_regularizer=reg,
        dilation_rate=dilation_rate,
        use_bias=use_bias,
    )(ip_)

    if batch_norm:
        conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm, name=name + "_bn_1")(conv)

    conv = tf.keras.layers.Activation("relu", name=name + "_act_1")(conv)

    conv = tf.keras.layers.SeparableConv2D(
        nfilters,
        (3, 3),
        padding="same",
        name=name + "_conv_2",
        use_bias=use_bias,
        dilation_rate=dilation_rate,
        depthwise_regularizer=reg,
        pointwise_regularizer=reg,
    )(conv)

    if batch_norm:
        conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm, name=name + "_bn_2")(conv)

    return tf.keras.layers.Activation("relu", name=name + "_act_2")(conv)


def build_unet_model_fun(x_init, weight_decay=0.05, batch_norm=True, final_activation="sigmoid"):

    axis_batch_norm = 3

    reg = tf.keras.regularizers.l2(weight_decay)

    conv1 = conv_blocks(x_init, 32, axis_batch_norm, reg, name="input", batch_norm=batch_norm)

    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_1")(conv1)

    conv2 = conv_blocks(pool1, 64, axis_batch_norm, reg, name="pool1", batch_norm=batch_norm)

    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pooling_2")(conv2)

    conv3 = conv_blocks(pool2, 128, axis_batch_norm, reg, name="pool2", batch_norm=batch_norm)

    up8 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(
                64, (2, 2), strides=(2, 2), padding="same", name="upconv1", kernel_regularizer=reg
            )(conv3),
            conv2,
        ],
        axis=axis_batch_norm,
        name="concatenate_up_1",
    )

    conv8 = conv_blocks(up8, 64, axis_batch_norm, reg, name="up1", batch_norm=batch_norm)

    up9 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(
                32, (2, 2), strides=(2, 2), padding="same", name="upconv2", kernel_regularizer=reg
            )(conv8),
            conv1,
        ],
        axis=axis_batch_norm,
        name="concatenate_up_2",
    )

    conv9 = conv_blocks(up9, 32, axis_batch_norm, reg, name="up2", batch_norm=batch_norm)

    conv10 = tf.keras.layers.Conv2D(
        1, (1, 1), kernel_regularizer=reg, name="linear_model", activation=final_activation
    )(conv9)

    return conv10


# NORM_OFF_PROBAV = np.array([0.43052389, 0.40560079, 0.46504526, 0.23876471])
# ID_KERNEL_INITIALIZER =np.eye(4)[None, None]
# c11.set_weights([ID_KERNEL_INITIALIZER, -NORM_OFF_PROBAV])


def load_model(shape=(None, None), bands_input=4, weight_decay=0.0, final_activation="sigmoid"):
    ip = tf.keras.layers.Input(shape + (bands_input,), name="ip_cloud")
    c11 = tf.keras.layers.Conv2D(bands_input, (1, 1), name="normalization_cloud", trainable=False)
    x_init = c11(ip)
    conv2d10 = build_unet_model_fun(
        x_init, weight_decay=weight_decay, final_activation=final_activation, batch_norm=True
    )
    return tf.keras.models.Model(inputs=[ip], outputs=[conv2d10], name="UNet-Clouds")
