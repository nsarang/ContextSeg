import keras.backend as K
from keras import layers, models


def ContextSeg(image_shape,
               n_labels):
    h, w, d = image_shape
    assert d == 3

    input_orig = layers.Input(shape=(h, w, d), name='input_orig')
    input_feats = layers.Input(shape=(h, w, 1), name='input_feats')
    input_scaled = layers.Input(shape=(h//4, w//4, d), name='input_scaled')

    merge_wfeats = layers.concatenate([input_orig, input_feats])

    output_sh = _shallow_branch(merge_wfeats)
    output_dp = _deep_branch(input_scaled)

    output = _fuse_branch(n_labels, output_sh, output_dp, input_feats)
    
    model = models.Model(inputs=[input_orig, input_feats, input_scaled],
                         outputs=output,
                         name='ContextSeg_model')
    return model


def _shallow_branch(img_input):
    x = _conv2d_block(img_input, filters=32, kernel=3, strides=1,
                      padding='same', name='sh_conv1')
    
    x = _sep_conv2d_block(x, filters=64, kernel=3, strides=2,
                          padding='same', name='sh_conv2')
    x = _sep_conv2d_block(x, filters=128, kernel=3, strides=2,
                          padding='same', name='sh_conv3')
    x = _sep_conv2d_block(x, filters=128, kernel=3, strides=2,
                          padding='same', name='sh_conv4')
     
    x = _conv2d_block(x, filters=256, kernel=3, strides=1,
                      padding='same', name='sh_conv5')
    return x


def _deep_branch(img_input, alpha=1):     
    first_block_filters = (32 * alpha + 4) // 8 * 8
    x = _conv2d_block(img_input, filters=first_block_filters, kernel=3,
                      strides=1, padding='same', name='dp_conv1')

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=1)

    x = _inverted_res_block(x, filters=48, alpha=alpha, stride=2,
                            expansion=6, block_id=2)
    x = _inverted_res_block(x, filters=48, alpha=alpha, stride=1,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=48, alpha=alpha, stride=1,
                            expansion=6, block_id=4)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=5)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=9)
    x = _inverted_res_block(x, filters=128, alpha=alpha, stride=1,
                            expansion=6, block_id=10)

    if alpha > 1.0:
        last_block_filters = (192 * alpha + 4) // 8 * 8
    else:
        last_block_filters = 192

    x = _conv2d_block(x, filters=last_block_filters, kernel=3,
                      strides=1, padding='same', name='dp_conv_last')
    return x


def _fuse_branch(n_labels, input_tensor1, input_tensor2, input_feats):
    x = _conv2d_block(input_tensor1, filters=128, kernel=1, # kernel=3
                      strides=1, padding='same', name='fs_sh_conv1')
    
    y = _deconv2d_block(input_tensor2, filters=256, kernel=3, strides=2,
                        padding='same', name='fs_dp_deconv1')
    y = _sep_conv2d_block(y, filters=128, kernel=3, strides=1, dilation_rate=4,
                          padding='same', name='fs_dp_conv1')
    y = _conv2d_block(y, filters=128, kernel=1, # kernel=3
                      strides=1, padding='same', name='fs_dp_conv2')
    
    merged = layers.add([x, y])
    merged = _inverted_res_block(merged, filters=128, alpha=1, stride=1,
                                 expansion=1, block_id=11)
    # merged = _sep_conv2d_block(merged, filters=192, kernel=3, strides=1,
    #                            dilation_rate=4, padding='same', name='fs_conv1')
    merged = _deconv2d_block(merged, filters=128, kernel=3, strides=2,
                             padding='same', name='fs_deconv1')
    # merged = _sep_conv2d_block(merged, filters=64, kernel=3, strides=1,
    #                            dilation_rate=4, padding='same', name='fs_conv2')
    merged = _inverted_res_block(merged, filters=128, alpha=1, stride=1,
                                 expansion=1, block_id=12)
    merged = _deconv2d_block(merged, filters=64, kernel=3, strides=2,
                             padding='same', name='fs_deconv2')
    merged = _inverted_res_block(merged, filters=64, alpha=1, stride=1,
                                 expansion=1, block_id=13)
    # merged = _sep_conv2d_block(merged, filters=64, kernel=3, strides=1,
    #                            dilation_rate=4, padding='same', name='fs_conv3')
    merged = _deconv2d_block(merged, filters=64, kernel=3, strides=2,
                             padding='same', name='fs_deconv3')

    merged = layers.concatenate([merged, input_feats])
    merged = _inverted_res_block(merged, filters=64, alpha=1, stride=1,
                                 expansion=6, block_id=14)

    merged = layers.Conv2D(n_labels,
                           kernel_size=3, # kernel=1
                           activation='softmax',
                           padding='same',
                           name='conv_last')(merged) 
    return merged


def _conv2d_block(x, filters, kernel, strides, padding, name):
    x = layers.Conv2D(filters,
                      kernel_size=kernel,
                      strides=strides,
                      padding=padding,
                      name=name)(x) 
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=name+'_BN')(x)
    x = layers.ReLU(6., name=name+'_relu')(x)
    return x


def _sep_conv2d_block(x, filters, kernel, strides, padding, name,
                      dilation_rate=1):
    name += '_depthwise'
    x = layers.SeparableConv2D(filters,
                               kernel_size=kernel,
                               strides=strides,
                               dilation_rate=dilation_rate,
                               padding=padding,
                               name=name)(x) 
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=name+'_BN')(x)
    x = layers.ReLU(6., name=name+'_relu')(x)
    return x


def _deconv2d_block(x, filters, kernel, strides, padding, name):
    x = layers.Conv2DTranspose(filters,
                               kernel_size=kernel,
                               strides=strides,
                               padding=padding,
                               name=name)(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=name+'_BN')(x)
    x = layers.ReLU(6., name=name+'_relu')(x)
    return x


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = -1
    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = (pointwise_conv_filters + 4) // 8 * 8;
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=_correct_pad(x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def _correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    input_size = K.int_shape(inputs)[1:3]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))