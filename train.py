#!/usr/bin/env python

import pickle, cv2
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from data_generator import Generator
from ssd.ssd_utils import BBoxUtility, MultiboxLoss, get_prior_boxes
from params import *
from ssd.net import *

model_type = sys.argv[1]
data_file, num_classes = 'train/IITB.pkl', 5
path_prefix = 'train/data/'
img_height, img_width = 300, 300
prior_box_configs = prior_box_configs_300
if model_type == '512':
    img_height, img_width = 512, 512
    prior_box_configs = prior_box_configs_512
variances = [0.1, 0.1, 0.2, 0.2]

prior_boxes = get_prior_boxes(img_width, img_height, prior_box_configs, variances)
#pickle.dump(prior_boxes, open('default_prior_boxes_{}x{}.pkl'.format(img_height, img_width), 'wb'))
bbox_util = BBoxUtility(num_classes, prior_boxes, use_tf=True)

data = pickle.load(open(data_file, 'rb'))
keys = data.keys()
num_train = int(round(0.8 * len(keys)))
train_keys, val_keys = keys[:num_train], keys[num_train:]

data_gen = Generator(data, bbox_util, 1, path_prefix, train_keys, val_keys, (img_height, img_width))

model = SSD300((img_height, img_width, 3), num_classes=num_classes)
if model_type == '512':
    model = SSD512((img_height, img_width, 3), num_classes=num_classes)
model.compile(optimizer=Adam(lr=3e-4), loss=MultiboxLoss(num_classes, neg_pos_ratio=3.0).compute_loss)
model.summary()

nb_epochs = 120
callbacks = [ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_weights_only=True),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)]
history = model.fit_generator(data_gen.generate(True), data_gen.train_batches,
                              epochs=nb_epochs, callbacks=callbacks,
                              validation_data=data_gen.generate(False),
                              validation_steps=data_gen.val_batches)

model.save('tf_VGG_VOC0712Plus_SSD_{}x{}.hdf5'.format(img_height, img_width))
