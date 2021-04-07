import grad_cam

import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf

test_images_dir = "E:/CR image/test_img/"

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# test
test_images_path = os.listdir(test_images_dir)
test_df = pd.DataFrame({'image_path':test_images_path})

test_gen = ImageDataGenerator(rescale=1.0/255.0)
test_gen = test_gen.flow_from_dataframe(test_df,
                    directory=test_images_dir,
                    x_col='image_path',
                    y_col=None,
                    class_mode=None,
                    target_size=(128, 128),
                    color_mode="rgb",
                    shuffle = False)


model = load_model('E:/CR image/classification_model_1_1.h5')

img_index = 16

image = cv2.imread(test_images_dir + test_df['image_path'][img_index])
image = cv2.resize(image, (128, 128))
image = image.astype('float32') / 255
image = np.expand_dims(image, axis=0)

predictions = model.predict(image)
predictions = np.argmax(predictions, axis=1)

icam = grad_cam.GradCAM(model, predictions, 'conv2d_2')
heatmap = icam.compute_heatmap(image)
heatmap = cv2.resize(heatmap, (32, 32))

image = cv2.imread(test_images_dir + test_df['image_path'][img_index])
image = cv2.resize(image, (32, 32))
print(heatmap.shape, image.shape)

(heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

fig, ax = plt.subplots(1, 3)
plt.title(predictions)
ax[0].imshow(heatmap)
ax[1].imshow(image)
ax[2].imshow(output)
plt.show()
