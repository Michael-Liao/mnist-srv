import os
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


class CovNet02(object):
    def __init__(self):
        img = Input(shape=(28, 28, 1))

        cnn = Conv2D(32, 5, padding='same', activation='relu')(img)
        cnn = MaxPooling2D(strides=2)(cnn)
        cnn = Conv2D(64, 5, padding='same', activation='relu')(cnn)
        cnn = MaxPooling2D(strides=2)(cnn)

        cnn = Flatten()(cnn)
        cnn = Dense(1024, activation='relu')(cnn)
        cnn = Dropout(0.4)(cnn)

        out = Dense(10, activation='softmax')(cnn)

        self.model = Model(inputs=img, outputs=out)
        self.model.compile(
            'sgd',
            loss='categorical_crossentropy',
            metrics=['acc']
        )
        self.model.summary()

    def train(self, training_gen, validation_gen):
        early = EarlyStopping(
            monitor='val_loss', mode='min', patience=3, min_delta=0.01
        )
        self.model.fit_generator(
            training_gen,
            validation_data=validation_gen,
            epochs=20,
            callbacks=[early],
            use_multiprocessing=True,
            workers=2
        )

        # save weights
        if not os.path.exists('./weights'):
            os.makedirs('./weights')

        self.model.save_weights('./weights/mnist.h5')

    def evaluate(self, evaluation_gen, weights='./weights/mnist.h5'):
        self.model.load_weights(weights)
        results = self.model.evaluate_generator(evaluation_gen)

        return results[1]*100   # accuracy in percents

    def train_one(self, x, y, validation_gen):
        self.model.load_weights('./weights/mnist.h5')
        self.model.fit(x, y, batch_size=1)

        # TODO: replace if has better accuracy

    def predict_one(self, x, weights='./weights/mnist.h5'):
        self.model.load_weights(weights)
        y = self.model.predict_on_batch(x)

        result = y[0].argmax()

        return result


def cov_net_02(weights='./weights/mnist.h5'):
    m = CovNet02()
    m.model.load_weights(weights)

    return m.model
