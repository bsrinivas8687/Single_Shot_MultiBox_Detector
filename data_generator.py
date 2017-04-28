import cv2
import numpy as np
from random import shuffle, randrange


class Generator(object):
    def __init__(self, gt, bbox_util, batch_size, path_prefix, train_keys, val_keys, image_size, save_dir=None,
                 saturation_prob=0.5, brightness_prob=0.5, contrast_prob=0.5, hue_prob=0.5, hflip_prob=0, vflip_prob=0):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.saturation_prob = saturation_prob
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.hue_prob = hue_prob
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.save_dir = save_dir

    def random_saturation(self, rgb):
        if np.random.random() < self.saturation_prob:
            delta = randrange(0.5, 1.5, _int = float)
            if abs(delta - 1.0) != 1e-03:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                channels = cv2.split(rgb)
                channels[1] *= delta
                rgb = cv2.merge(channels)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_HSV2RGB)
        return rgb

    def random_brightness(self, rgb):
        if np.random.random() < self.brightness_prob:
            delta = randrange(-32, 32, _int = float)
            if abs(delta) > 0:
                rgb += delta
        return rgb

    def random_contrast(self, rgb):
        if np.random.random() < self.contrast_prob:
            delta = randrange(0.5, 1.5, _int = float)
            if abs(delta - 1.0) > 1e-03:
                rgb *= delta
        return rgb

    def random_hue(self, rgb):
        if np.random.random() < self.hue_prob:
            delta = randrange(-18, 18, _int = float)
            if abs(delta) > 0:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                channels = cv2.split(rgb)
                channels[0] += delta
                rgb = cv2.merge(channels)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_HSV2RGB)
        return rgb

    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def generate(self, train=True):
        image_count = 0
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:
                img_path = self.path_prefix + key
                img_path = img_path.split('.')[0] + '.jpg'
                img = cv2.imread(img_path)
                img = img.astype(np.float32)
                y = self.gt[key].copy()
                img = cv2.resize(img, self.image_size)
                img[:, :, 0] -= 104.0
                img[:, :, 1] -= 117.0
                img[:, :, 2] -= 124.0
                if train:
                    if np.random.random() < 0.5:
                        img = self.random_brightness(img)
                        img = self.random_contrast(img)
                        img = self.random_saturation(img)
                        img = self.random_hue(img)
                    else:
                        img = self.random_brightness(img)
                        img = self.random_saturation(img)
                        img = self.random_hue(img)
                        img = self.random_contrast(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if self.save_dir is not None and train:
                    image_name = self.save_dir + key + '_' + str(image_count) + '.jpg'
                    cv2.imwrite(image_name, img)
                    image_count += 1
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets
