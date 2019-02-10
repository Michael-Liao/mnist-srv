from sklearn.model_selection import train_test_split
import os
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from skimage.io import imread
from skimage import img_as_float
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

''' model creation '''
# IMAGE_SHAPE = (28, 28, 1)
img = Input(shape=(28, 28, 1))

cnn = Conv2D(32, 5, padding='same', activation='relu')(img)
cnn = MaxPooling2D(strides=2)(cnn)
cnn = Conv2D(64, 5, padding='same', activation='relu')(cnn)
cnn = MaxPooling2D(strides=2)(cnn)

cnn = Flatten()(cnn)
cnn = Dense(1024, activation='relu')(cnn)
cnn = Dropout(0.4)(cnn)

out = Dense(10, activation='softmax')(cnn)

model = Model(inputs=img, outputs=out)
model.compile('sgd', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

''' loading data '''

# paths = []
# for root, dirs, files in os.walk('./data/training'):
# #   print(root)
#   for name in files:
#     paths.append(os.path.join(root, name))
#     # print(name)
#   # for name in dirs:
#   #   print(root, name)
# filepath = paths[0]
# print(filepath)
# print(int(filepath.split('/')[-2]))


class DataProcessor(Sequence):
    def __init__(self, paths, batch_size=8, augmentation=None, shuffle=True):
        self.paths = paths
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.paths[idx*self.batch_size: (idx+1)*self.batch_size]

        batch_images = np.empty((len(batch_data), 28, 28, 1), dtype=float)
        batch_labels = np.empty(len(batch_data))
        for i, path in enumerate(batch_data):
            img = imread(path, as_gray=True)
            img = img_as_float(img)
            # CNN requires channel
            batch_images[i, ] = np.expand_dims(img, axis=2)
            batch_labels[i] = int(path.split('/')[-2])

        return batch_images, to_categorical(batch_labels, num_classes=10)
        # return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.paths)


# read in all image filenames in training directory
file_list = []
for root, _, files in os.walk('./data/training'):
    for fname in files:
        filepath = os.path.join(root, fname)
        file_list.append(filepath)

# split
train_list, valid_list = train_test_split(
    file_list, test_size=0.1, shuffle=True)
# use generator
training_generator = DataProcessor(train_list)
validation_generator = DataProcessor(valid_list)

# dataIt = iter(training_generator)
# for i in range(2):
#     x, y = next(dataIt)
#     print(x[0])

''' training '''
early = EarlyStopping(monitor='val_loss', mode='min',
                      patience=3, min_delta=0.01)
model.fit_generator(
    training_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early],
    use_multiprocessing=True,
    workers=2
)

# save weights
if not os.path.exists('./weights'):
    os.makedirs('./weights')

model.save_weights('./weights/mnist.h5')

''' test (eval) '''
test_files = []
for root, _, files in os.walk('./data/testing'):
    for fname in files:
        filepath = os.path.join(root, fname)
        test_files.append(filepath)

test_generator = DataProcessor(test_files)

result = model.evaluate_generator(test_generator)

print(result)
