import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import tensorflow as tf

test_images_dir = "E:/CR image/test_img/"
sample_sub_path = "E:/CR image/test_500.csv"

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# test
test_images_path = os.listdir(test_images_dir)
# test_df = pd.DataFrame({'image_path':test_images_path})
test_df = pd.read_csv(sample_sub_path)

test_gen = ImageDataGenerator(rescale=1.0/255.0)
test_gen = test_gen.flow_from_dataframe(test_df,
                    directory=test_images_dir,
                    x_col='image_path',
                    y_col='name',
                    class_mode ='categorical',
                    target_size=(128, 128),
                    color_mode="rgb",
                    shuffle = False)

# 查看one hot encoding結果
print(test_gen.class_indices)
# {'N': 0, 'P': 1}

model = load_model('E:/CR image/classification_model_1_1.h5')

def num2PN(num):
  if num == 1:
    return 'P'
  else:
    return 'N'



loss, accuracy = model.evaluate(test_gen)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

predictions = model.predict(test_gen)
predictions = np.argmax(predictions, axis=1)

for i, file in enumerate(test_df['image_path'][:12]):
    img = imread(test_images_dir+file)
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    print(file)
    plt.title(f"Class:{num2PN(predictions[i])}")
    # plt.title(f"Class:{predictions[i]}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

for i, file in enumerate(test_df['image_path'][12:24]):
    img = imread(test_images_dir+file)
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    print(file)
    plt.title(f"Class:{num2PN(predictions[i])}")
    # plt.title(f"Class:{predictions[i]}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

