#! /user/bin/env python3

import keras
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, concatenate, Dropout

import pickle
import numpy as np
from pathlib import Path

from image.enhancement import filters
from image import feature_extraction, utils

class EmotionsModel(object):

    def __init__(
        self,
        verbose=False,
        create_new=False,
        use_hog=None,
        use_cnn=None,
        use_lm=None,
        emotions=None,
    ):
        # pathes constants
        self.model_path = './saved-models/emotions_model.f5'
        self.model_specs_path = './saved-models/emotions_model_specs.bin'


        self.verbose = verbose
        
        # model numbers
        self.imageSize = 150
        self.batch_size = 128
        self.epochs = 50


        if not create_new and self.has_saved_model():
            self.load_model()
            self.is_trained = True
        else:
            if use_hog == None or use_cnn == None or use_lm == None or emotions == None:
                raise Exception(
                    'When creating new model, all model specs (use_hog, use_cnn, use_lm, emotions) must be given'
                )

            # set model specs
            self.use_hog = use_hog
            self.use_cnn = use_cnn
            self.use_lm = use_lm
            self.emotions = emotions

            self.model = self.__create_model__()
            self.is_trained = False

    def fit(self, xs, ys, should_save_model=True, epochs_num=None):
        if epochs_num == None:
            epochs_num = self.epochs

        xs = self.__transform_input__(xs)
        ys = to_categorical(ys, len(self.emotions))

        history = self.model.fit(
            xs, ys,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
            verbose=self.verbose,
        )

        self.is_trained = True
        if should_save_model:
            self.save_model()

        return history

    def test(self, faces, targets):
        if not self.is_trained:
            raise Exception("Model not trained yet")

        faces = self.__transform_input__(faces)
        targets = to_categorical(targets, len(self.emotions))
        return self.model.evaluate(faces, targets)

    def predict(self, faces, prob_emotion=False):
        if not self.is_trained:
            raise Exception("Model not trained yet")

        is_one_face = False
        if type(faces) is not list:
            faces = [faces]
            is_one_face = True

        faces = self.__transform_input__(faces)

        res = self.model.predict(faces, batch_size=self.batch_size).tolist()

        if not prob_emotion:
            for i, all in enumerate(res):
                res[i] = self.emotions[np.argmax(all)]

        if is_one_face:
            res = res[0]

        return res

    def __transform_input__(self, images, should_enhance=True):
        lms, hogs, imgs = [], [], []

        if self.verbose:
            size = 30
            print('Extracting Features -   0% [', end='')
            for i in range(size):
                print('.', end='')
            print(']', end='')

        for i, image in enumerate(images):
            img = self.__enhance_image__(image) if should_enhance else image
            if self.use_cnn:
                imgs.append(np.reshape(img, (self.imageSize, self.imageSize, 1)))
            if self.use_lm:
                lms.append(feature_extraction.get_face_landmarks(img))
            if self.use_hog:
                hogs.append(feature_extraction.sk_get_hog(img, pixels_per_cell=(30, 30)))

            # printing the done percentage
            if self.verbose:
                done = (i+1) / len(images)
                print('\rExtracting Features - %3d%% [' % (done*100), end='')
                for j in range(size):
                    cnt = int(done*size)
                    if j <= cnt:
                        print('=', end='')
                    elif j == cnt+1:
                        print('>', end='')
                    else:
                        print('.', end='')
                print(']', end='')
        if self.verbose:
            print()

        ret = []
        if self.use_cnn:
            ret.append(np.array(imgs))
        if self.use_hog:
            ret.append(np.array(hogs))
        if self.use_lm:
            ret.append(np.array(lms))

        return ret

    def __enhance_image__(self, img):
        # normalize image to wanted size
        img = utils.normalize_image(img, self.imageSize)

        # remove salt and pepper
        img = filters.median(img)

        # sharpen images
        img = filters.laplacian(img)

        return img

    def save_model(self):
        # save model weights
        self.model.save(self.model_path)

        # save model specs
        with open(self.model_specs_path, 'wb') as specsFile:
            pickle.dump(
                (
                    self.use_hog,
                    self.use_cnn,
                    self.use_lm,
                    self.emotions,
                ),
                specsFile
            )

    def load_model(self):
        # load model weights
        self.model = keras.models.load_model(self.model_path)

        # load model specs
        with open(self.model_specs_path, 'rb') as specsFile:
            self.use_hog, self.use_cnn, self.use_lm, self.emotions = pickle.load(specsFile)

    def has_saved_model(self):
        model_path = Path(self.model_path)
        model_specs_path = Path(self.model_specs_path)
        return model_path.is_file() and model_specs_path.is_file()

    def __create_model__(self):
        conv_activation = 'sigmoid'
        dense_activation = 'sigmoid'
        keep_prob = 0.956

        # ========================== CNN Part ==========================
        if self.use_cnn:
            input_image = Input(batch_shape=(None, 150, 150, 1), dtype='float32', name='input_image')
            x = BatchNormalization(axis=-1)(input_image)

            x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=conv_activation)(input_image)
            x = BatchNormalization(axis=-1)(x)

            x = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=conv_activation)(x)
            x = BatchNormalization(axis=-1)(x)

            x = MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(x)

            x = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation=conv_activation)(x)
            x = MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(x)
            x = BatchNormalization(axis=-1)(x)

            x = Flatten()(x)
            x = Dense(units=2048, activation=dense_activation)(x)
            x = Dropout(rate=keep_prob)(x)
            x = Dense(units=1024, activation=dense_activation)(x)
            x = Dropout(rate=keep_prob)(x)

            x = BatchNormalization(axis=-1)(x)

            outputCNN = x

        # ========================== Image features part ==========================
        if self.use_lm or self.use_hog:

            if self.use_lm:
                inputLandmarks = Input(batch_shape=(None, 68, 2), dtype='float32', name='input_landmarks')
                flatLandmarks = Flatten()(inputLandmarks)
                normalizedLandmarks = BatchNormalization(axis=-1)(flatLandmarks)
                outputImage = normalizedLandmarks

            if self.use_hog:
                inputHOG = Input(batch_shape=(None, 512),dtype='float32', name='input_HOG')
                normalizedHog = BatchNormalization(axis=-1)(inputHOG)
                outputImage = normalizedHog

            if self.use_lm and self.use_hog:
                outputImage = concatenate([normalizedHog, normalizedLandmarks])

            outputImage = Dense(
                units=32, activation=dense_activation)(outputImage)
            outputImage = BatchNormalization(axis=-1)(outputImage)

            outputImage = Dense(
                units=128, activation=dense_activation)(outputImage)
            outputImage = BatchNormalization(axis=-1)(outputImage)

        if (self.use_lm or self.use_hog) and self.use_cnn:
            concat_output = concatenate([outputCNN, outputImage])
        elif self.use_cnn:
            concat_output = outputCNN
        else:
            concat_output = outputImage

        output = Dense(units=128, activation=dense_activation)(concat_output)
        output = BatchNormalization(axis=-1)(output)

        output = Dense(units=512, activation=dense_activation)(output)
        output = BatchNormalization(axis=-1)(output)

        output = Dense(units=len(self.emotions), activation='softmax')(concat_output)

        # ========================== Model creating part ==========================
        inputs = []
        if self.use_cnn:
            inputs.append(input_image)
        if self.use_hog:
            inputs.append(inputHOG)
        if self.use_lm:
            inputs.append(inputLandmarks)

        model = Model(inputs=inputs, outputs=[output])

        model.compile(
            optimizer='adadelta',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
