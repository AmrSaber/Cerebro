#! /user/bin/env python3

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.models import Model


def create_model():

	conv_activation = 'relu'
	dense_activation = 'relu'

	# first dimension of shape was None
	input_image = Input(batch_shape=(None, 48, 48, 1), dtype='float32', name='input_image')

	x = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation=conv_activation)(input_image)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(x)
	x = BatchNormalization(axis=-1)(x)

	x = Conv2D(filters=96, kernel_size=(5, 5), padding='same', activation=conv_activation)(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(x)
	x = BatchNormalization(axis=-1)(x)

	x = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation=conv_activation)(x)
	x = MaxPooling2D(pool_size=(2, 2), strides=2, data_format="channels_last")(x)
	# x = BatchNormalization(axis=-1)(x)

	x = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation=conv_activation)(x)
	# x = BatchNormalization(axis=-1)(x)

	x = Flatten()(x)

	x = Dense(units=2048, activation=dense_activation)(x)

	output = Dense(units=5, activation='softmax')(x)

	model = Model(inputs=[input_image], outputs=[output])
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

	return model
