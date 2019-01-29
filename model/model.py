#! /user/bin/env python3

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, concatenate
from keras.models import Model

#FIXME: pass HOG and LandMarks to the model in fit, test and predict
class EmotionsModel(object):

	def __init__(self, targets_count, create_new=False):
		if not create_new and self.has_saved_model():
			self.model = self.load_model()
			self.is_trained = True
		else:
			self.model = self._create_model(targets_count)
			self.is_trained = False

	def fit(self, xs, ys, should_save_model=True):
		# try with 13 epochs as the github example
		history = self.model.fit(xs, ys, epochs=1)

		self.is_trained = True
		if should_save_model: self.save_model()

		return history

	def test(self, faces, targets):
		if not self.is_trained: raise Exception("Model not trained yet")
		return self.model.evaluate(faces, targets)

	def predict(self, faces):
		if not self.is_trained: raise Exception("Model not trained yet")
		pass

	# TODO
	def save_model(self):
		pass

	# TODO
	def load_model(self):
		pass

	# TODO
	def has_saved_model(self):
		return False

	def _create_model(self, targets_count):
		conv_activation = 'relu'
		dense_activation = 'relu'

		# ========================== CNN Part ==========================
		input_image = Input(batch_shape=(None, 48, 48, 1), dtype='float32', name='input_image')

		x = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation=conv_activation)(input_image)
		x = MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(x)
		x = BatchNormalization(axis=-1)(x)

		x = Conv2D(filters=96, kernel_size=(5, 5), padding='same', activation=conv_activation)(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(x)
		x = BatchNormalization(axis=-1)(x)

		x = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation=conv_activation)(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(x)

		x = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation=conv_activation)(x)

		x = Flatten()(x)

		outputCNN = Dense(units=2048, activation=dense_activation)(x)

		# ========================== Image features part ==========================
		inputHOG = Input(batch_shape=(None, 8), dtype='float32', name='input_HOG')

		#TODO: Edit this shape when LM function is done
		inputLandmarks = Input(batch_shape=(None, 64, 2), dtype='float32', name='input_landmarks')
		flatLandmarks = Flatten()(inputLandmarks)

		mergeImage = concatenate([inputHOG , flatLandmarks])

		outputImage = Dense(units=128, activation=dense_activation)(mergeImage)

		finalMerge = concatenate([outputCNN, outputImage])

		finalDense = Dense(units=1024, activation= dense_activation)(finalMerge)

		output = Dense(units=targets_count, activation='softmax')(finalDense)


		model = Model(inputs=[input_image, inputHOG, inputLandmarks], outputs=[output])
		model.compile(optimizer='adam', loss='categorical_crossentropy', learning_rate=0.016, metrics=['accuracy'])
		# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

		return model
