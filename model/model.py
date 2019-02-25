#! /user/bin/env python3

import sys; sys.path.insert(1, '../image')
import feature_extraction
from enhancement import filters

# TODO: resolve the dependeny on reader (when the model is done)
from reader import emotions

from pathlib import Path

import numpy as np

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, concatenate ,Dropout
from keras.models import Model
from keras.optimizers import SGD

class EmotionsModel(object):

	def __init__(self, targets_count, create_new=False, use_hog=False):
		self.model_path = '../saved-models/emotions_model.f5'
		self.use_hog = use_hog
		self.batch_size = 128
		self.epochs = 2
		if not create_new and self.has_saved_model():
			self.load_model()
			self.is_trained = True
		else:
			self.model = self.__create_model__(targets_count)
			self.is_trained = False

	def fit(self, xs, ys, should_save_model=True, epochs_num=None):
		if epochs_num == None: epochs_num = self.epochs
		xs = self.__transform_input__(xs)

		history = self.model.fit(xs, ys, batch_size=self.batch_size, epochs=epochs_num)

		self.is_trained = True
		if should_save_model: self.save_model()

		return history

	def test(self, faces, targets):
		if not self.is_trained: raise Exception("Model not trained yet")
		faces = self.__transform_input__(faces)
		return self.model.evaluate(faces, targets)

	def predict(self, faces, prob_emotion=False):
		if not self.is_trained: raise Exception("Model not trained yet")

		is_one_face = False
		if type(faces) is not list:
			faces = [faces]
			is_one_face = True

		faces = self.__transform_input__(faces)

		res = self.model.predict(faces, batch_size=self.batch_size)

		if not prob_emotion:
			for i, all from enumerate(res):
				res[i] = emotions[np.argmax(all)]

		if is_one_face: res = res[0]

		return res

	def __transform_input__(self, images):
		lms, hogs, imgs = [], [], []
		for image in images:
			img = self.__enhance_image__(image)
			imgs.append(np.reshape(img, (48, 48, 1)))
			lms.append(feature_extraction.get_face_landmarks(img))
			if self.use_hog: hogs.append(feature_extraction.sk_get_hog(img))

		ret = [np.array(imgs), np.array(lms)]
		if self.use_hog: ret.insert(1, np.array(hogs))

		return ret

	def __enhance_image__(self, img):
		# remove salt and peper
		img = filters.median(img)

		# remove gaussian noise
		img = filters.fastNLMeans(img)

		# sharpen images
		# img = filters.laplacian(img)

		# remove noise resulting from laplacian
		# img = filters.median(img)

		return img

	def save_model(self):
		self.model.save(self.model_path)

	def load_model(self):
		self.model = keras.models.load_model(self.model_path)

	def has_saved_model(self):
		model_path = Path(self.model_path)
		return model_path.is_file()

	def __create_model__(self, targets_count):
		conv_activation = 'relu'
		dense_activation = 'relu'
		keep_prob = 0.956

		# ========================== CNN Part ==========================
		input_image = Input(batch_shape=(None, 48, 48, 1), dtype='float32', name='input_image')
		x = BatchNormalization(axis=-1)(input_image)

		x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=conv_activation)(input_image)
		x = BatchNormalization(axis=-1)(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last")(x)

		x = Conv2D(filters=128, kernel_size=(3, 3), padding='valid', activation=conv_activation)(x)
		x = BatchNormalization(axis=-1)(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last")(x)

		x = Conv2D(filters=256, kernel_size=(3, 3), padding='valid', activation=conv_activation)(x)
		x = BatchNormalization(axis=-1)(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=2, data_format="channels_last")(x)

		x = Flatten()(x)

		x = Dropout(rate=keep_prob)(x)
		x = Dense(units=2048 , activation=dense_activation)(x)
		x = BatchNormalization(axis=-1)(x)

		x = Dropout(rate=keep_prob)(x)
		x = Dense(units=1024 , activation=dense_activation)(x)
		x = BatchNormalization(axis=-1)(x)

		outputCNN = x

		# ========================== Image features part ==========================
		inputLandmarks = Input(batch_shape=(None, 68, 2), dtype='float32', name='input_landmarks')
		flatLandmarks = Flatten()(inputLandmarks)
		normalizedLandmarks = BatchNormalization(axis=-1)(flatLandmarks)

		if self.use_hog:
			inputHOG = Input(batch_shape=(None, 128), dtype='float32', name='input_HOG')
			normalizedHog = BatchNormalization(axis=-1)(inputHOG)

			outputImage = concatenate([normalizedHog, normalizedLandmarks])
		else:
			outputImage = normalizedLandmarks

		# outputImage = BatchNormalization(axis=-1)(outputImage)

		outputImage = Dense(units=1024, activation=dense_activation)(outputImage)
		outputImage = BatchNormalization(axis=-1)(outputImage)

		concat_output = concatenate([outputCNN, outputImage])
		# concat_output = BatchNormalization(axis=-1)(concat_output)

		output = Dense(units=256 ,activation=dense_activation)(concat_output)
		output = BatchNormalization(axis=-1)(output)

		output = Dense(units=targets_count, activation='softmax')(concat_output)

		# ========================== Model creating part ==========================
		inputs = [input_image, inputLandmarks]
		if self.use_hog: inputs.insert(1, inputHOG)

		model = Model(inputs=inputs, outputs=[output])

		# sgd = SGD(lr=0.016, decay=0.864, momentum=0.95, nesterov=True)
		model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

		return model
