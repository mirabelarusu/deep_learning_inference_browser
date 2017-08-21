""" For helper code """

import numpy as np

from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import core
from keras.models import Model
from keras.optimizers import Adam, Adamax, SGD

from keras import backend as K
K.set_image_dim_ordering('tf')


def dice_coef(y_true, y_pred, smooth = 1. ):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	return coef

def dice_coef2(y_true, y_pred, smooth = 1. ):
	diff1 = K.flatten(K.round(y_pred[:,:,:,2,:]) - 
		K.round(y_pred[:,:,:,0,:])*K.round(y_pred[:,:,:,2,:]))
	diff2 = K.flatten(K.round(y_pred[:,:,:,1,:]) - 
		K.round(y_pred[:,:,:,2,:])*K.round(y_pred[:,:,:,1,:]))
	diff3 = K.flatten(K.round(y_pred[:,:,:,1,:]) - 
		K.round(y_pred[:,:,:,0,:])*K.round(y_pred[:,:,:,1,:]))
	#diff3 = K.flatten(y_pred[:,:,:,1,:] - y_pred[:,:,:,0,:]*y_pred[:,:,:,1,:])

	coef1 = 1.0/(K.sum(diff1) + smooth)
	coef2 = 1.0/(K.sum(diff2) + smooth)
	coef3 = 1.0/(K.sum(diff2) + smooth)

	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	coef = (2. * intersection + smooth) / \
		(K.sum(y_true_f) + K.sum(y_pred_f) + smooth+K.sum(diff1)+K.sum(diff2)+K.sum(diff3))
	return coef
#	return (coef+ coef1 + coef2 + coef3)/4.0

def dice_coef3(y_true, y_pred, smooth = 1. ):
	coef3 = K.mean(K.square((y_pred - y_true)), axis=-1)
	y_pred = K.clip(y_pred,0,1)
	diff = K.sum(y_pred, axis = 3)-1
	tr = K.sum(y_true, axis = 3)
	diff_f = K.square(K.flatten(diff*tr))
	diff1 = K.sum(diff_f)/(48.0*48.0)
	print (y_pred.get_shape(), diff1.get_shape(), diff1) 
	weight = y_true*99.0 + 1.
	coef2 = K.mean(K.square((y_pred - y_true)*weight), axis=-1)
		
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	#print (y_pred._keras_shape, diff1.get_shape(), coef.get_shape()) 
	#return (diff1+coef2)
	return -K.log(coef)+0.0*diff1
	#return -K.log(coef3)


def dice_coef_loss(y_true, y_pred):
	return -K.log(dice_coef(y_true, y_pred))

def dice_coef_loss2(y_true, y_pred):
	return -(dice_coef(y_true, y_pred))

def acc(y_true, y_pred, smooth = 1. ):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f == y_pred_f)
	acc = (intersection) / (K.sum(y_true_f>=0) + K.sum(y_pred_f>=0))
	return acc

def acc_loss(y_true, y_pred):
	return -acc(y_true, y_pred)

def binary_crossentropy2(y_true, y_pred):
	print K.print_tensor(y_true,'CrossEntropy2'), K.print_tensor(y_pred)
	return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def mean_squared_error2(y_true, y_pred, weight=None):
	weight = y_true*99.0 + 1.
	return K.mean(K.square((y_pred - y_true)*weight), axis=-1)

def mean_squared_error3(y_true, y_pred, weight=None):
	w =  (1.0/0.000060, 1.0/0.000728, 1.0/0.000150, 1.0/0.000184, 1.0/1.0)

	weight = np.ones(y_pred._keras_shape[1:])
	weight[:,:,0] = w(0)/np.sum(w)
	weight[:,:,1] = w(1)/np.sum(w)
	weight[:,:,2] = w(2)/np.sum(w)
	weight[:,:,3] = w(3)/np.sum(w)
	print "Shape tensor", y_pred._keras_shape[1:], np.min(weight), np.max(weight)
	M = max(weight)*(-K.mean(y_pred,axis=3)+np.ones(y_pred._keras_shape[1:3]))
	N = K.mean(K.square((y_pred - y_true)*weight), axis=-1)
	print M.get_shape(), N.get_shape()
	return M+N 
	#np.max(weight)*K.mean((-K.mean(y_pred,axis=3)+np.ones(y_pred._keras_shape[1:3])), axis=-1)


def model9(weights=False, summary=False, filepath=""):
	img_rows = 128
	img_cols = 128

	inputs = Input((1, img_rows, img_cols))
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

	up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
	conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
	conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

	up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
	conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
	conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

	up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
	conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
	conv8 = Dropout(0.5)(conv8)
	conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

	up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
	conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
	conv9 = Dropout(0.5)(conv9)
	conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

	conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

	model = Model(input=inputs, output=conv10)

	if weights and len(filepath)>0:
		model.load_weights(filepath)

	if summary:
		print(model.summary())

	return model

def model9TF(weights=False, filepath="", img_rows = 224, img_cols = 224):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	inputs = Input((img_rows, img_cols,1))
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
	conv8 = Dropout(0.5)(conv8)
	conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
	
	up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
	conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
	conv9 = Dropout(0.5)(conv9)
	conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

	conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
	model = Model(input=inputs, output=conv10)

	sgd = SGD(lr=1e-7)

	model.compile(loss=mean_squared_error2,
		optimizer=Adam(),
		metrics=['accuracy','fmeasure', dice_coef])

	"""
	#model.compile(optimizer=Adam(lr=1e-1), loss=dice_coef_loss, metrics=[dice_coef,'accuracy'])
	#model.compile(optimizer='rmsprop', loss=dice_coef_loss, metrics=[dice_coef])


	model.compile(loss='binary_crossentropy',
	#model.compile(loss='mean_squared_error',
		optimizer='sgd',
		#optimizer='rmsprop',
		metrics=['accuracy','fmeasure'])
	"""

	if weights and len(filepath)>0:
		model.load_weights(filepath)

	return model

def model9TF3(weights=False, filepath="", img_rows = 224, img_cols = 224, n_cl=3, 
	dropout=0.5):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	inputs = Input((img_rows, img_cols,n_cl))
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

	conv10 = Convolution2D(n_cl, 1, 1, activation='sigmoid')(conv9)
	model = Model(input=inputs, output=conv10)

	model.compile(loss=mean_squared_error2,
		optimizer=Adam(),
		metrics=['accuracy','fmeasure'])


	if weights and len(filepath)>0:
		model.load_weights(filepath)

	return model

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
	#model.compile(optimizer=Adam(),
		loss=dice_coef_loss,
		metrics=['accuracy','fmeasure'])

	if weights and len(filepath)>0:
		model.load_weights(filepath)

	if print_summary:
		print (model.summary())	

	return model

def model5_MultiLayer_5D(weights=False, filepath="", img_rows = 224, img_cols = 224, n_cl_in=3, n_cl_out=3, dropout=0.5, print_summary = False ):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	inputs = Input((img_rows, img_cols,n_cl_in,1))
	conv1 = Convolution3D(32, 3, 3, 1, activation='relu', border_mode='same')(inputs)
	conv1 = Convolution3D(32, 3, 3, 1, activation='relu', border_mode='same')(conv1)
	pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

	conv2 = Convolution3D(64, 3, 3, 1, activation='relu', border_mode='same')(pool1)
	conv2 = Convolution3D(64, 3, 3, 1, activation='relu', border_mode='same')(conv2)
	pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

	conv3 = Convolution3D(128, 3, 3, 1, activation='relu', border_mode='same')(pool2)
	conv3 = Convolution3D(128, 3, 3, 1, activation='relu', border_mode='same')(conv3)
	pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

	conv4 = Convolution3D(256, 3, 3, 1, activation='relu', border_mode='same')(pool3)
	conv4 = Convolution3D(256, 3, 3, 1, activation='relu', border_mode='same')(conv4)
	pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)

	conv5 = Convolution3D(512, 3, 3, 1, activation='relu', border_mode='same')(pool4)
	conv5 = Convolution3D(512, 3, 3, 1, activation='relu', border_mode='same')(conv5)

	up6 = merge([UpSampling3D(size=(2, 2, 1))(conv5), conv4], mode='concat', concat_axis=4)
	conv6 = Convolution3D(256, 3, 3, 1, activation='relu', border_mode='same')(up6)
	conv6 = Convolution3D(256, 3, 3, 1, activation='relu', border_mode='same')(conv6)

	up7 = merge([UpSampling3D(size=(2, 2, 1))(conv6), conv3], mode='concat', concat_axis=4)
	conv7 = Convolution3D(128, 3, 3, 1, activation='relu', border_mode='same')(up7)
	conv7 = Convolution3D(128, 3, 3, 1, activation='relu', border_mode='same')(conv7)

	up8 = merge([UpSampling3D(size=(2, 2, 1))(conv7), conv2], mode='concat', concat_axis=4)
	conv8 = Convolution3D(64, 3, 3, 1, activation='relu', border_mode='same')(up8)
	conv8 = Dropout(dropout)(conv8)
	conv8 = Convolution3D(64, 3, 3, 1, activation='relu', border_mode='same')(conv8)
	
	up9 = merge([UpSampling3D(size=(2, 2, 1))(conv8), conv1], mode='concat', concat_axis=4)
	conv9 = Convolution3D(32, 3, 3, 1, activation='relu', border_mode='same')(up9)
	conv9 = Dropout(dropout)(conv9)
	conv9 = Convolution3D(32, 3, 3, 1, activation='relu', border_mode='same')(conv9)

	conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv9)
	model = Model(input=inputs, output=conv10)

	model.compile(\
		#optimizer=Adam(lr=0.001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.5),
		optimizer=Adam(),
		#optimizer='sgd',
		loss=dice_coef_loss,
		#loss='binary_crossentropy',
		#loss=mean_squared_error2,
		#loss='mean_squared_error',
		#loss='mean_absolute_error',
		metrics=['accuracy','fmeasure'])

	if weights and len(filepath)>0:
		model.load_weights(filepath)

	if print_summary: 
		print (model.summary())	

	return model

def model5_MultiLayer_3D(weights=False, filepath="", img_rows = 224, img_cols = 224, n_cl_in=3,n_cl_out=3, dropout=0.2):
	""" difference from model: img_rows and cols, order of axis, and concat_axis"""
	
	inputs = Input((img_rows, img_cols,n_cl_in,1))
	conv1 = Convolution3D(32, 3, 3, n_cl_in, activation='relu', border_mode='same')(inputs)
	conv1 = Convolution3D(32, 3, 3, n_cl_in, activation='relu', border_mode='same')(conv1)
	pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

	conv2 = Convolution3D(64, 3, 3, n_cl_in, activation='relu', border_mode='same')(pool1)
	conv2 = Convolution3D(64, 3, 3, n_cl_in, activation='relu', border_mode='same')(conv2)
	pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

	conv3 = Convolution3D(128, 3, 3, n_cl_in, activation='relu', border_mode='same')(pool2)
	conv3 = Convolution3D(128, 3, 3, n_cl_in, activation='relu', border_mode='same')(conv3)
	pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

	conv4 = Convolution3D(256, 3, 3, n_cl_in, activation='relu', border_mode='same')(pool3)
	conv4 = Convolution3D(256, 3, 3, n_cl_in, activation='relu', border_mode='same')(conv4)
	pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)

	conv5 = Convolution3D(512, 3, 3, n_cl_in, activation='relu', border_mode='same')(pool4)
	conv5 = Convolution3D(512, 3, 3, n_cl_in, activation='relu', border_mode='same')(conv5)

	up6 = merge([UpSampling3D(size=(2, 2, 1))(conv5), conv4], mode='concat', concat_axis=4)
	conv6 = Convolution3D(256, 3, 3, n_cl_in, activation='relu', border_mode='same')(up6)
	conv6 = Convolution3D(256, 3, 3, n_cl_in, activation='relu', border_mode='same')(conv6)

	up7 = merge([UpSampling3D(size=(2, 2, 1))(conv6), conv3], mode='concat', concat_axis=4)
	conv7 = Convolution3D(128, 3, 3, n_cl_in, activation='relu', border_mode='same')(up7)
	conv7 = Convolution3D(128, 3, 3, n_cl_in, activation='relu', border_mode='same')(conv7)

	up8 = merge([UpSampling3D(size=(2, 2, 1))(conv7), conv2], mode='concat', concat_axis=4)
	conv8 = Convolution3D(64, 3, 3, n_cl_in, activation='relu', border_mode='same')(up8)
	conv8 = Dropout(dropout)(conv8)
	conv8 = Convolution3D(64, 3, 3, n_cl_in, activation='relu', border_mode='same')(conv8)
	
	up9 = merge([UpSampling3D(size=(2, 2, 1))(conv8), conv1], mode='concat', concat_axis=4)
	conv9 = Convolution3D(32, 3, 3, n_cl_in, activation='relu', border_mode='same')(up9)
	conv9 = Dropout(dropout)(conv9)
	conv9 = Convolution3D(32, 3, 3, n_cl_in, activation='relu', border_mode='same')(conv9)

	conv10 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv9)
	model = Model(input=inputs, output=conv10)

	model.compile(optimizer=Adam(),
		loss=dice_coef_loss,
		metrics=['accuracy','fmeasure'])

	if weights and len(filepath)>0:
		model.load_weights(filepath)

	print (model.summary())	

	return model

