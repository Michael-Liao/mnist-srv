import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from skimage.io import imread
from skimage import img_as_float


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
        if self.shuffle is True:
            np.random.shuffle(self.paths)
