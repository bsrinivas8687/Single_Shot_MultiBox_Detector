import os, cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from ssd.ssd_utils import BBoxUtility
from ssd.net import SSD300, SSD512


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
np.set_printoptions(suppress=True)


model_type = sys.argv[1]
base_path = 'test_pics/'
confidence = 0.6
classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
           'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'DiningTable',
           'Dog', 'Horse', 'Motorbike', 'Person', 'PottedPlant',
           'Sheep', 'Sofa', 'Train', 'TV/Monitor', 'Background']
num_classes = len(classes)
img_height, img_width = 300, 300
model = SSD300((img_height, img_width, 3), num_classes=num_classes)
if model_type == '512':
    img_height, img_width = 512, 512
    model = SSD512((img_height, img_width, 3), num_classes=num_classes)

model.summary()
model.load_weights('weights/tf_VGG_VOC0712Plus_SSD_{}x{}.hdf5'.format(img_height, img_width))
bbox_util = BBoxUtility(num_classes, session=K.get_session(), use_tf=True)

inputs = []
images = []
image_names = os.listdir(base_path)
for image_name in image_names:
    image = cv2.imread(base_path + image_name)
    images.append(image[:, :, ::-1])
    image = image.astype(np.float32)
    image = cv2.resize(image, (img_height, img_width))
    image[:, :, 0] -= 104.0
    image[:, :, 1] -= 117.0
    image[:, :, 2] -= 124.0
    inputs.append(image)
inputs = np.asarray(inputs)

predictions = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(predictions)


for i, img in enumerate(images):
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    top_indices = [i for i, conf in enumerate(det_conf) if conf >= confidence]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    plt.imshow(img / 255.0)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    plt.show()
