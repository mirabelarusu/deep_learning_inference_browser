""" For helper code """

import numpy as np

from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import core
from keras.models import Model
from keras.optimizers import Adam

from keras import backend as K
K.set_image_dim_ordering('tf')

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))


def dice_coef(y_true, y_pred, smooth = 1. ):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	return coef

def dice_coef_loss(y_true, y_pred):
	return -K.log(dice_coef(y_true, y_pred))

def model5_MultiLayer(weights=False, 
	filepath="", 
	img_rows = 224, 
	img_cols = 224, 
	n_cl_in=3,
	n_cl_out=3, 
	dropout=0.2, 
	learning_rate = 0.001,
	print_summary = False):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	inputs = Input((img_rows, img_cols,n_cl_in))
	conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
	conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
	conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
	conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
	conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
	conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

	up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
	conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
	conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

	up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
	conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
	conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

	up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
	conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
	conv8 = Dropout(dropout)(conv8)
	conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
	
	up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
	conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
	conv9 = Dropout(dropout)(conv9)
	conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

	conv10 = Convolution2D(n_cl_out, 1, 1, activation='sigmoid')(conv9)
	model = Model(input=inputs, output=conv10)

	
	model.compile(optimizer=Adam(lr=learning_rate),
		loss=dice_coef_loss,
		metrics=['accuracy'])

	if weights and len(filepath)>0:
		model.load_weights(filepath)

	if print_summary:
		print (model.summary())	

	return model
