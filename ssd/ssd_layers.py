import numpy as np
from keras.layers import Layer, InputSpec
from keras import backend as K


class Normalize(Layer):
    def __init__(self, scale, **kwargs):
        self.axis = 3 if K.image_dim_ordering() == 'tf' else 1
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.W_shape = (input_shape[self.axis],)
        gamma = np.ones(self.W_shape) * self.scale
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        x_normed = K.l2_normalize(x, self.axis)
        x_normed *= self.gamma
        return x_normed

    def get_config(self):
        config = { 'scale': self.scale, }
        base_config = super(Normalize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PriorBox(Layer):
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[1.0], clip=False, **kwargs):
        if min_size <= 0:
            raise Exception('min_size must be positive.')
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
        if K.image_dim_ordering() == 'tf':
            self.haxis = 1
            self.waxis = 2
        else:
            self.haxis = 2
            self.waxis = 3
        self.img_size = img_size
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0, 1.0]
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = clip
        self.flip = flip
        super(PriorBox, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        num_priors = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors * layer_width * layer_height
        return (input_shape[0], num_boxes, 8)

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
            input_shape = K.int_shape(x)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        img_width = self.img_size[0]
        img_height = self.img_size[1]
        num_priors = len(self.aspect_ratios)

        step_x = float(img_width) / float(layer_width)
        step_y = float(img_height) / float(layer_height)

        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)

        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)

        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))

        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        if self.clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        prior_boxes_tensor = K.expand_dims(K.variable(prior_boxes), 0)
        pattern = [K.shape(x)[0], 1, 1]
        prior_boxes_tensor = K.tile(prior_boxes_tensor, pattern)
        return prior_boxes_tensor

    def get_config(self):
        config = {
            'img_size': tuple(self.img_size),
            'min_size': self.min_size,
            'max_size': self.max_size,
            'aspect_ratios': tuple(self.aspect_ratios),
            'variances': tuple(self.variances),
            'flip': self.flip,
            'clip': self.clip }
        base_config = super(PriorBox, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))