from keras.models import Model
from keras.layers import Input, ZeroPadding2D, Conv2D, MaxPool2D
from keras.layers import concatenate, Flatten, Reshape, Activation
from ssd_layers import PriorBox, Normalize


def SSD300(input_shape=(300, 300, 3), num_classes=21):

    net = {}
    net['inputs'] = Input(shape=input_shape, name='inputs')

    net['conv1_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv1_1_zp')(net['inputs'])
    net['conv1_1'] = Conv2D(64, (3, 3), activation='relu', strides=(1, 1), name='conv1_1')(net['conv1_1_zp'])
    net['conv1_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv1_2_zp')(net['conv1_1'])
    net['conv1_2'] = Conv2D(64, (3, 3), activation='relu', strides=(1, 1), name='conv1_2')(net['conv1_2_zp'])
    net['pool1'] = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(net['conv1_2'])

    net['conv2_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv2_1_zp')(net['pool1'])
    net['conv2_1'] = Conv2D(128, (3, 3), activation='relu', strides=(1, 1), name='conv2_1')(net['conv2_1_zp'])
    net['conv2_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv2_2_zp')(net['conv2_1'])
    net['conv2_2'] = Conv2D(128, (3, 3), activation='relu', strides=(1, 1), name='conv2_2')(net['conv2_2_zp'])
    net['pool2'] = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(net['conv2_2'])

    net['conv3_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv3_1_zp')(net['pool2'])
    net['conv3_1'] = Conv2D(256, (3, 3), activation='relu', strides=(1, 1), name='conv3_1')(net['conv3_1_zp'])
    net['conv3_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv3_2_zp')(net['conv3_1'])
    net['conv3_2'] = Conv2D(256, (3, 3), activation='relu', strides=(1, 1), name='conv3_2')(net['conv3_2_zp'])
    net['conv3_3_zp'] = ZeroPadding2D(padding=(1, 1), name='conv3_3_zp')(net['conv3_2'])
    net['conv3_3'] = Conv2D(256, (3, 3), activation='relu', strides=(1, 1), name='conv3_3')(net['conv3_3_zp'])
    net['pool3'] = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(net['conv3_3'])

    net['conv4_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_1_zp')(net['pool3'])
    net['conv4_1'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv4_1')(net['conv4_1_zp'])
    net['conv4_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_2_zp')(net['conv4_1'])
    net['conv4_2'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv4_2')(net['conv4_2_zp'])
    net['conv4_3_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_3_zp')(net['conv4_2'])
    net['conv4_3'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv4_3')(net['conv4_3_zp'])
    net['pool4'] = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool4')(net['conv4_3'])

    net['conv5_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv5_1_zp')(net['pool4'])
    net['conv5_1'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv5_1')(net['conv5_1_zp'])
    net['conv5_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv5_2_zp')(net['conv5_1'])
    net['conv5_2'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv5_2')(net['conv5_2_zp'])
    net['conv5_3_zp'] = ZeroPadding2D(padding=(1, 1), name='conv5_3_zp')(net['conv5_2'])
    net['conv5_3'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv5_3')(net['conv5_3_zp'])
    net['pool5_zp'] = ZeroPadding2D(padding=(1, 1), name='pool5_zp')(net['conv5_3'])
    net['pool5'] = MaxPool2D(pool_size=(3, 3), strides=(1, 1), name='pool5')(net['pool5_zp'])

    net['fc6_zp'] = ZeroPadding2D(padding=(6, 6), name='fc6_zp')(net['pool5'])
    net['fc6'] = Conv2D(1024, (3, 3), activation='relu', strides=(1, 1), dilation_rate=(6, 6), name='fc6')(net['fc6_zp'])

    net['fc7'] = Conv2D(1024, (1, 1), activation='relu', strides=(1, 1), name='fc7')(net['fc6'])

    net['conv6_1'] = Conv2D(256, (1, 1), activation='relu', strides=(1, 1), name='conv6_1')(net['fc7'])
    net['conv6_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv6_2_zp')(net['conv6_1'])
    net['conv6_2'] = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), name='conv6_2')(net['conv6_2_zp'])

    net['conv7_1'] = Conv2D(128, (1, 1), activation='relu', strides=(1, 1), name='conv7_1')(net['conv6_2'])
    net['conv7_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv7_2_zp')(net['conv7_1'])
    net['conv7_2'] = Conv2D(256, (3, 3), activation='relu', strides=(2, 2), name='conv7_2')(net['conv7_2_zp'])

    net['conv8_1'] = Conv2D(128, (1, 1), activation='relu', strides=(1, 1), name='conv8_1')(net['conv7_2'])
    net['conv8_2'] = Conv2D(256, (3, 3), activation='relu', strides=(1, 1), name='conv8_2')(net['conv8_1'])

    net['conv9_1'] = Conv2D(128, (1, 1), activation='relu', strides=(1, 1), name='conv9_1')(net['conv8_2'])
    net['conv9_2'] = Conv2D(256, (3, 3), activation='relu', strides=(1, 1), name='conv9_2')(net['conv9_1'])

    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])

    num_priors = 4
    net['conv4_3_norm_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_3_norm_mbox_loc_zp')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv4_3_norm_mbox_loc')(net['conv4_3_norm_mbox_loc_zp'])
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    net['conv4_3_norm_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_3_norm_mbox_conf_zp')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv4_3_norm_mbox_conf')(net['conv4_3_norm_mbox_conf_zp'])
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])
    net['conv4_3_norm_mbox_priorbox'] = PriorBox((300, 300), min_size=30.0, max_size=60.0,
                                                 aspect_ratios=[2.0], variances=[0.10, 0.10, 0.20, 0.20],
                                                 flip=True, clip=False, name='conv4_3_norm_mbox_priorbox')(net['conv4_3_norm'])

    num_priors = 6
    net['fc7_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='fc7_mbox_loc_zp')(net['fc7'])
    net['fc7_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), activation='relu', strides=(1, 1), name='fc7_mbox_loc')(net['fc7_mbox_loc_zp'])
    net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    net['fc7_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='fc7_mbox_conf_zp')(net['fc7'])
    net['fc7_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), activation='relu', strides=(1, 1), name='fc7_mbox_conf')(net['fc7_mbox_conf_zp'])
    net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])
    net['fc7_mbox_priorbox'] = PriorBox((300, 300), min_size=60.0, max_size=111.0,
                                        aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],
                                        flip=True, clip=False, name='fc7_mbox_priorbox')(net['fc7'])

    net['conv6_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv6_2_mbox_loc_zp')(net['conv6_2'])
    net['conv6_2_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv6_2_mbox_loc')(net['conv6_2_mbox_loc_zp'])
    net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    net['conv6_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv6_2_mbox_conf_zp')(net['conv6_2'])
    net['conv6_2_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv6_2_mbox_conf')(net['conv6_2_mbox_conf_zp'])
    net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])
    net['conv6_2_mbox_priorbox'] = PriorBox((300, 300), min_size=111.0, max_size=162.0,
                                            aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],
                                            flip=True, clip=False, name='conv6_2_mbox_priorbox')(net['conv6_2'])

    net['conv7_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv7_2_mbox_loc_zp')(net['conv7_2'])
    net['conv7_2_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv7_2_mbox_loc')(net['conv7_2_mbox_loc_zp'])
    net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    net['conv7_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv7_2_mbox_conf_zp')(net['conv7_2'])
    net['conv7_2_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv7_2_mbox_conf')(net['conv7_2_mbox_conf_zp'])
    net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])
    net['conv7_2_mbox_priorbox'] = PriorBox((300, 300), min_size=162.0, max_size=213.0,
                                            aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],
                                            flip=True, clip=False, name='conv7_2_mbox_priorbox')(net['conv7_2'])

    num_priors = 4
    net['conv8_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv8_2_mbox_loc_zp')(net['conv8_2'])
    net['conv8_2_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv8_2_mbox_loc')(net['conv8_2_mbox_loc_zp'])
    net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    net['conv8_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv8_2_mbox_conf_zp')(net['conv8_2'])
    net['conv8_2_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv8_2_mbox_conf')(net['conv8_2_mbox_conf_zp'])
    net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])
    net['conv8_2_mbox_priorbox'] = PriorBox((300, 300), min_size=213.0, max_size=264.0,
                                            aspect_ratios=[2.0], variances=[0.10, 0.10, 0.20, 0.20],
                                            flip=True, clip=False, name='conv8_2_mbox_priorbox')(net['conv8_2'])

    net['conv9_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv9_2_mbox_loc_zp')(net['conv9_2'])
    net['conv9_2_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv9_2_mbox_loc')(net['conv9_2_mbox_loc_zp'])
    net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    net['conv9_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv9_2_mbox_conf_zp')(net['conv9_2'])
    net['conv9_2_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), activation='relu', strides=(1, 1), name='conv9_2_mbox_conf')(net['conv9_2_mbox_conf_zp'])
    net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])
    net['conv9_2_mbox_priorbox'] = PriorBox((300, 300), min_size=264.0, max_size=315.0,
                                            aspect_ratios=[2.0], variances=[0.10, 0.10, 0.20, 0.20],
                                            flip=True, clip=False, name='conv9_2_mbox_priorbox')(net['conv9_2'])

    net['mbox_loc'] = concatenate(inputs=[net['conv4_3_norm_mbox_loc_flat'],
                                          net['fc7_mbox_loc_flat'],
                                          net['conv6_2_mbox_loc_flat'],
                                          net['conv7_2_mbox_loc_flat'],
                                          net['conv8_2_mbox_loc_flat'],
                                          net['conv9_2_mbox_loc_flat']], axis=1, name='mbox_loc')
    net['mbox_conf'] = concatenate(inputs=[net['conv4_3_norm_mbox_conf_flat'],
                                           net['fc7_mbox_conf_flat'],
                                           net['conv6_2_mbox_conf_flat'],
                                           net['conv7_2_mbox_conf_flat'],
                                           net['conv8_2_mbox_conf_flat'],
                                           net['conv9_2_mbox_conf_flat']], axis=1, name='mbox_conf')
    net['mbox_priorbox'] = concatenate(inputs=[net['conv4_3_norm_mbox_priorbox'],
                                               net['fc7_mbox_priorbox'],
                                               net['conv6_2_mbox_priorbox'],
                                               net['conv7_2_mbox_priorbox'],
                                               net['conv8_2_mbox_priorbox'],
                                               net['conv9_2_mbox_priorbox']], axis=1, name='mbox_priorbox')

    num_boxes = net['mbox_loc']._keras_shape[-1] // 4

    net['mbox_conf_reshape'] = Reshape(target_shape=(num_boxes, num_classes), name='mbox_conf_reshape')(net['mbox_conf'])
    net['mbox_conf_softmax'] = Activation('softmax', name='mbox_conf_softmax')(net['mbox_conf_reshape'])
    net['mbox_loc_reshape'] = Reshape(target_shape=(num_boxes, 4), name='mbox_loc_reshape')(net['mbox_loc'])
    net['detection_out'] = concatenate(inputs=[net['mbox_loc_reshape'],
                                               net['mbox_conf_softmax'],
                                               net['mbox_priorbox']], axis=2, name='detection_out')
    model = Model(net['inputs'], net['detection_out'])
    return model

def SSD512(input_shape=(512, 512, 3), num_classes=21):

    net = {}
    net['inputs'] = Input(shape=input_shape, name='inputs')

    net['conv1_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv1_1_zp')(net['inputs'])
    net['conv1_1'] = Conv2D(64, (3, 3), activation='relu', strides=(1, 1), name='conv1_1')(net['conv1_1_zp'])
    net['conv1_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv1_2_zp')(net['conv1_1'])
    net['conv1_2'] = Conv2D(64, (3, 3), activation='relu', strides=(1, 1), name='conv1_2')(net['conv1_2_zp'])
    net['pool1'] = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(net['conv1_2'])

    net['conv2_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv2_1_zp')(net['pool1'])
    net['conv2_1'] = Conv2D(128, (3, 3), activation='relu', strides=(1, 1), name='conv2_1')(net['conv2_1_zp'])
    net['conv2_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv2_2_zp')(net['conv2_1'])
    net['conv2_2'] = Conv2D(128, (3, 3), activation='relu', strides=(1, 1), name='conv2_2')(net['conv2_2_zp'])
    net['pool2'] = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(net['conv2_2'])

    net['conv3_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv3_1_zp')(net['pool2'])
    net['conv3_1'] = Conv2D(256, (3, 3), activation='relu', strides=(1, 1), name='conv3_1')(net['conv3_1_zp'])
    net['conv3_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv3_2_zp')(net['conv3_1'])
    net['conv3_2'] = Conv2D(256, (3, 3), activation='relu', strides=(1, 1), name='conv3_2')(net['conv3_2_zp'])
    net['conv3_3_zp'] = ZeroPadding2D(padding=(1, 1), name='conv3_3_zp')(net['conv3_2'])
    net['conv3_3'] = Conv2D(256, (3, 3), activation='relu', strides=(1, 1), name='conv3_3')(net['conv3_3_zp'])
    net['pool3'] = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(net['conv3_3'])

    net['conv4_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_1_zp')(net['pool3'])
    net['conv4_1'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv4_1')(net['conv4_1_zp'])
    net['conv4_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_2_zp')(net['conv4_1'])
    net['conv4_2'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv4_2')(net['conv4_2_zp'])
    net['conv4_3_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_3_zp')(net['conv4_2'])
    net['conv4_3'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv4_3')(net['conv4_3_zp'])
    net['pool4'] = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool4')(net['conv4_3'])

    net['conv5_1_zp'] = ZeroPadding2D(padding=(1, 1), name='conv5_1_zp')(net['pool4'])
    net['conv5_1'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv5_1')(net['conv5_1_zp'])
    net['conv5_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv5_2_zp')(net['conv5_1'])
    net['conv5_2'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv5_2')(net['conv5_2_zp'])
    net['conv5_3_zp'] = ZeroPadding2D(padding=(1, 1), name='conv5_3_zp')(net['conv5_2'])
    net['conv5_3'] = Conv2D(512, (3, 3), activation='relu', strides=(1, 1), name='conv5_3')(net['conv5_3_zp'])
    net['pool5_zp'] = ZeroPadding2D(padding=(1, 1), name='pool5_zp')(net['conv5_3'])
    net['pool5'] = MaxPool2D(pool_size=(3, 3), strides=(1, 1), name='pool5')(net['pool5_zp'])

    net['fc6_zp'] = ZeroPadding2D(padding=(6, 6), name='fc6_zp')(net['pool5'])
    net['fc6'] = Conv2D(1024, (3, 3), activation='relu', strides=(1, 1), dilation_rate=(6, 6), name='fc6')(net['fc6_zp'])

    net['fc7'] = Conv2D(1024, (1, 1), activation='relu', strides=(1, 1), name='fc7')(net['fc6'])

    net['conv6_1'] = Conv2D(256, (1, 1), activation='relu', strides=(1, 1), name='conv6_1')(net['fc7'])
    net['conv6_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv6_2_zp')(net['conv6_1'])
    net['conv6_2'] = Conv2D(512, (3, 3), activation='relu', strides=(2, 2), name='conv6_2')(net['conv6_2_zp'])

    net['conv7_1'] = Conv2D(128, (1, 1), activation='relu', strides=(1, 1), name='conv7_1')(net['conv6_2'])
    net['conv7_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv7_2_zp')(net['conv7_1'])
    net['conv7_2'] = Conv2D(256, (3, 3), activation='relu', strides=(2, 2), name='conv7_2')(net['conv7_2_zp'])

    net['conv8_1'] = Conv2D(128, (1, 1), activation='relu', strides=(1, 1), name='conv8_1')(net['conv7_2'])
    net['conv8_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv8_2_zp')(net['conv8_1'])
    net['conv8_2'] = Conv2D(256, (3, 3), activation='relu', strides=(2, 2), name='conv8_2')(net['conv8_2_zp'])

    net['conv9_1'] = Conv2D(128, (1, 1), activation='relu', strides=(1, 1), name='conv9_1')(net['conv8_2'])
    net['conv9_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv9_2_zp')(net['conv9_1'])
    net['conv9_2'] = Conv2D(256, (3, 3), activation='relu', strides=(2, 2), name='conv9_2')(net['conv9_2_zp'])

    net['conv10_1'] = Conv2D(128, (1, 1), activation='relu', strides=(1, 1), name='conv10_1')(net['conv9_2'])
    net['conv10_2_zp'] = ZeroPadding2D(padding=(1, 1), name='conv10_2_zp')(net['conv10_1'])
    net['conv10_2'] = Conv2D(256, (4, 4), activation='relu', strides=(1, 1), name='conv10_2')(net['conv10_2_zp'])

    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])

    num_priors = 4
    net['conv4_3_norm_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_3_norm_mbox_loc_zp')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv4_3_norm_mbox_loc')(net['conv4_3_norm_mbox_loc_zp'])
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat')(net['conv4_3_norm_mbox_loc'])
    net['conv4_3_norm_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv4_3_norm_mbox_conf_zp')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv4_3_norm_mbox_conf')(net['conv4_3_norm_mbox_conf_zp'])
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='conv4_3_norm_mbox_conf_flat')(net['conv4_3_norm_mbox_conf'])
    net['conv4_3_norm_mbox_priorbox'] = PriorBox((512, 512), min_size=35.84, max_size=76.80,
                                                 aspect_ratios=[2.0], variances=[0.10, 0.10, 0.20, 0.20],
                                                 flip=True, clip=False, name='conv4_3_norm_mbox_priorbox')(net['conv4_3_norm'])

    num_priors = 6
    net['fc7_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='fc7_mbox_loc_zp')(net['fc7'])
    net['fc7_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), strides=(1, 1), name='fc7_mbox_loc')(net['fc7_mbox_loc_zp'])
    net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat')(net['fc7_mbox_loc'])
    net['fc7_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='fc7_mbox_conf_zp')(net['fc7'])
    net['fc7_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='fc7_mbox_conf')(net['fc7_mbox_conf_zp'])
    net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat')(net['fc7_mbox_conf'])
    net['fc7_mbox_priorbox'] = PriorBox((512, 512), min_size=76.80, max_size=153.60,
                                        aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],
                                        flip=True, clip=False, name='fc7_mbox_priorbox')(net['fc7'])

    net['conv6_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv6_2_mbox_loc_zp')(net['conv6_2'])
    net['conv6_2_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv6_2_mbox_loc')(net['conv6_2_mbox_loc_zp'])
    net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat')(net['conv6_2_mbox_loc'])
    net['conv6_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv6_2_mbox_conf_zp')(net['conv6_2'])
    net['conv6_2_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv6_2_mbox_conf')(net['conv6_2_mbox_conf_zp'])
    net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat')(net['conv6_2_mbox_conf'])
    net['conv6_2_mbox_priorbox'] = PriorBox((512, 512), min_size=153.60, max_size=230.40,
                                            aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],
                                            flip=True, clip=False, name='conv6_2_mbox_priorbox')(net['conv6_2'])

    net['conv7_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv7_2_mbox_loc_zp')(net['conv7_2'])
    net['conv7_2_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv7_2_mbox_loc')(net['conv7_2_mbox_loc_zp'])
    net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat')(net['conv7_2_mbox_loc'])
    net['conv7_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv7_2_mbox_conf_zp')(net['conv7_2'])
    net['conv7_2_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv7_2_mbox_conf')(net['conv7_2_mbox_conf_zp'])
    net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat')(net['conv7_2_mbox_conf'])
    net['conv7_2_mbox_priorbox'] = PriorBox((512, 512), min_size=230.40, max_size=307.20,
                                            aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],
                                            flip=True, clip=False, name='conv7_2_mbox_priorbox')(net['conv7_2'])

    net['conv8_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv8_2_mbox_loc_zp')(net['conv8_2'])
    net['conv8_2_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv8_2_mbox_loc')(net['conv8_2_mbox_loc_zp'])
    net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat')(net['conv8_2_mbox_loc'])
    net['conv8_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv8_2_mbox_conf_zp')(net['conv8_2'])
    net['conv8_2_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv8_2_mbox_conf')(net['conv8_2_mbox_conf_zp'])
    net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat')(net['conv8_2_mbox_conf'])
    net['conv8_2_mbox_priorbox'] = PriorBox((512, 512), min_size=307.20, max_size=384.00,
                                            aspect_ratios=[2.0, 3.0], variances=[0.10, 0.10, 0.20, 0.20],
                                            flip=True, clip=False, name='conv8_2_mbox_priorbox')(net['conv8_2'])

    num_priors = 4
    net['conv9_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv9_2_mbox_loc_zp')(net['conv9_2'])
    net['conv9_2_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv9_2_mbox_loc')(net['conv9_2_mbox_loc_zp'])
    net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat')(net['conv9_2_mbox_loc'])
    net['conv9_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv9_2_mbox_conf_zp')(net['conv9_2'])
    net['conv9_2_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv9_2_mbox_conf')(net['conv9_2_mbox_conf_zp'])
    net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat')(net['conv9_2_mbox_conf'])
    net['conv9_2_mbox_priorbox'] = PriorBox((512, 512), min_size=384.00, max_size=460.80,
                                            aspect_ratios=[2.0], variances=[0.10, 0.10, 0.20, 0.20],
                                            flip=True, clip=False, name='conv9_2_mbox_priorbox')(net['conv9_2'])

    net['conv10_2_mbox_loc_zp'] = ZeroPadding2D(padding=(1, 1), name='conv10_2_mbox_loc_zp')(net['conv10_2'])
    net['conv10_2_mbox_loc'] = Conv2D(4 * num_priors, (3, 3), strides=(1, 1), name='conv10_2_mbox_loc')(net['conv10_2_mbox_loc_zp'])
    net['conv10_2_mbox_loc_flat'] = Flatten(name='conv10_2_mbox_loc_flat')(net['conv10_2_mbox_loc'])
    net['conv10_2_mbox_conf_zp'] = ZeroPadding2D(padding=(1, 1), name='conv10_2_mbox_conf_zp')(net['conv10_2'])
    net['conv10_2_mbox_conf'] = Conv2D(num_classes * num_priors, (3, 3), strides=(1, 1), name='conv10_2_mbox_conf')(net['conv10_2_mbox_conf_zp'])
    net['conv10_2_mbox_conf_flat'] = Flatten(name='conv10_2_mbox_conf_flat')(net['conv10_2_mbox_conf'])
    net['conv10_2_mbox_priorbox'] = PriorBox((512, 512), min_size=460.80, max_size=537.60,
                                             aspect_ratios=[2.0], variances=[0.10, 0.10, 0.20, 0.20],
                                             flip=True, clip=False, name='conv10_2_mbox_priorbox')(net['conv10_2'])

    net['mbox_loc'] = concatenate(inputs=[net['conv4_3_norm_mbox_loc_flat'],
                                          net['fc7_mbox_loc_flat'],
                                          net['conv6_2_mbox_loc_flat'],
                                          net['conv7_2_mbox_loc_flat'],
                                          net['conv8_2_mbox_loc_flat'],
                                          net['conv9_2_mbox_loc_flat'],
                                          net['conv10_2_mbox_loc_flat']], axis=1, name='mbox_loc')
    net['mbox_conf'] = concatenate(inputs=[net['conv4_3_norm_mbox_conf_flat'],
                                           net['fc7_mbox_conf_flat'],
                                           net['conv6_2_mbox_conf_flat'],
                                           net['conv7_2_mbox_conf_flat'],
                                           net['conv8_2_mbox_conf_flat'],
                                           net['conv9_2_mbox_conf_flat'],
                                           net['conv10_2_mbox_conf_flat']], axis=1, name='mbox_conf')
    net['mbox_priorbox'] = concatenate(inputs=[net['conv4_3_norm_mbox_priorbox'],
                                               net['fc7_mbox_priorbox'],
                                               net['conv6_2_mbox_priorbox'],
                                               net['conv7_2_mbox_priorbox'],
                                               net['conv8_2_mbox_priorbox'],
                                               net['conv9_2_mbox_priorbox'],
                                               net['conv10_2_mbox_priorbox']], axis=1, name='mbox_priorbox')

    num_boxes = net['mbox_loc']._keras_shape[-1] // 4

    net['mbox_conf_reshape'] = Reshape(target_shape=(num_boxes, num_classes), name='mbox_conf_reshape')(net['mbox_conf'])
    net['mbox_conf_softmax'] = Activation('softmax', name='mbox_conf_softmax')(net['mbox_conf_reshape'])
    net['mbox_loc_reshape'] = Reshape(target_shape=(num_boxes, 4), name='mbox_loc_reshape')(net['mbox_loc'])
    net['detection_out'] = concatenate(inputs=[net['mbox_loc_reshape'],
                                               net['mbox_conf_softmax'],
                                               net['mbox_priorbox']], axis=2, name='detection_out')
    model = Model(net['inputs'], net['detection_out'])
    return model
