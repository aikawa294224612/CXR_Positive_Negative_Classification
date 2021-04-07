import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# 因為資料不平均，所以做一筆一圈學

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

test_images_dir = "E:/CR image/test_img/"
train_images_dir = "E:/CR image/train_img/"
sample_sub_path = "E:/CR image/training_list_40_order.csv"
epochs = 15

# test
test_images_path = os.listdir(test_images_dir)
test_df = pd.DataFrame({'image_path':test_images_path})

df = pd.read_csv(sample_sub_path)

train_df, valid_df = train_test_split(df, test_size = 0.15)
print("Training set:", train_df.shape)
print("Validation set:", valid_df.shape)

# 建議xray不翻轉，水平移即可
datagen = ImageDataGenerator(
    # rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = ImageDataGenerator(rescale = 1.0/255.0,
                # horizontal_flip = True,
                # vertical_flip   = True,
                fill_mode = 'nearest',
                # rotation_range = 10,
                width_shift_range = 0.2,
                height_shift_range= 0.2,
                shear_range= 0.15,
                brightness_range= (.5,1.2),
                zoom_range = 0.2)

train_gen = train_gen.flow_from_dataframe(train_df,
                      directory = train_images_dir,
                      x_col = 'image_path',
                      y_col = 'name',
                      target_size =(128, 128),
                      class_mode = 'categorical',  # one hot encoded
                      batch_size = 32,
                      color_mode = 'rgb',
                      shuffle = True)

valid_gen = ImageDataGenerator(rescale=1./255)
valid_gen = valid_gen.flow_from_dataframe(valid_df,
                      directory = train_images_dir,
                      x_col='image_path',
                      y_col='name',
                      target_size =(128, 128),
                      class_mode='categorical',
                      batch_size=32)

test_gen = ImageDataGenerator(rescale=1.0/255.0)
test_gen = test_gen.flow_from_dataframe(test_df,
                    directory=test_images_dir,
                    x_col='image_path',
                    y_col=None,
                    class_mode=None,
                    target_size=(128, 128),
                    color_mode="rgb",
                    shuffle = False)

def showImage(image_gen):
    for x_gens, y_gens in image_gen:
        print(x_gens.shape)
        x_gen_shape = x_gens.shape[1:]
        i = 0
        for sample_img, sample_class in zip(x_gens, y_gens):
            plt.subplot(2,4,i+1)
            plt.title(f'Class:{np.argmax(sample_class)}')
            plt.axis('off')
            plt.imshow(sample_img)

            i += 1

            if i >= 8:
                break
        break

    plt.show()

showImage(train_gen)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, input_shape=(128, 128, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 做early stopping
earlystop = EarlyStopping(monitor='val_accuracy',
                          patience=2)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                      patience=2,
                      verbose=1,
                      factor=0.5,
                      min_lr=0.0001)


history = model.fit(train_gen,
          # steps_per_epoch = len(train_df)//32,
          epochs = epochs,
          validation_data = valid_gen,
          # validation_steps = len(valid_df)//BATCH_SIZE
          )

model.save('E:/CR image/classification_model_1_1.h5')

print(history.history.keys())
print("Acc:", history.history['accuracy'][-1])
print("Val Acc:", history.history['val_accuracy'][-1])

def show_train_history(train_history, train, validation, title):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(title)
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(history, 'accuracy', 'val_accuracy', 'Train History')
show_train_history(history, 'loss', 'val_loss', 'Loss History')

def num2PN(num):
  if num == 1:
    return 'Positive'
  else:
    return 'Negtive'


predictions = model.predict(test_gen)
predictions = np.argmax(predictions, axis=1)


for i, file in enumerate(test_df['image_path'][:12]):
    img = imread(test_images_dir+file)
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.title(f"Class:{num2PN(predictions[i])}")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()



