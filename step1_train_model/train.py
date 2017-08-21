""" Train keras model using unet architecture """

from keras import backend as K
K.set_image_dim_ordering('tf')	
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, Callback
from keras.metrics import binary_accuracy
from keras.callbacks import History 
from keras.models import Model
K.set_image_dim_ordering('tf')	# Theano dimension ordering in this code

import numpy as np
import os
import cv2
from sklearn.metrics import accuracy_score
import SimpleITK as sitk

from step0_preprocess import * 
from helper import * 
import settings

#all the time the same random seed
#FIXME: Doest seem to be working, the results are not reproducible
#np.random.seed(123123123)
#import tensorflow as tf
#tf.set_random_seed(123123123)

class customCallback(Callback):
	"""	Allows to do quick tests during training

	# Arguments
		validation_data: np.array containing both images and predictions
	"""
	def __init__(self, validation_data = None, path_out = None):
		self.validation_data	= validation_data
		self.path_out			= path_out

	def on_train_begin(self, logs={}):
		self.losses = []
		#run the end 
		self.on_epoch_end(None, logs)

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

		# end of epoch evaluation
		print ("")
		msk = self.model.predict(self.validation_data[0], verbose=1)

		radioLab = np.sum(np.where(msk>0))/float(np.sum(np.where(msk>=0)))
		minSl = np.min(msk)
		maxSl = np.max(msk)
		tf_session = K.get_session()
		accK  = binary_accuracy(msk, self.validation_data[1].astype('float32')).eval(
			session=tf_session)
		accNp = accuracy_score(msk.flatten().astype('uint8'), 
			self.validation_data[1].flatten().astype('uint8'))
		dice = dice_coef(msk,self.validation_data[1].astype('float32')).eval(session=tf_session)

		print("  {:6.3g} {:6.3g} {:6.3g} {:6.3g} {:6.3g} {:6.3g}".format(minSl,maxSl, 
			radioLab, accK, accNp, dice))

		for predLab in range(msk.shape[len(msk.shape)-1]):
			sl = msk[:,:,:,predLab]
			gt = self.validation_data[1][:,:,:,predLab]
			radioLab = np.sum(np.where(sl>0))/ float(np.sum(np.where(sl>=0)))
			minSl = np.min(sl)
			maxSl = np.max(sl)

			accK  = binary_accuracy(sl, gt.astype('float32')).eval(
				session=tf_session)
			accNp = accuracy_score(sl.flatten().astype('uint8'), 
				gt.flatten().astype('uint8'))
			dice = dice_coef(sl,gt.astype('float32')).eval(session=tf_session)
			print("{:d} {:6.3g} {:6.3g} {:6.3g} {:6.3g} {:6.3g} {:6.3g}".format(predLab, 
				minSl, maxSl, radioLab, accK, accNp, dice))


		try:

			layer_name = 'convolution2d_18'
			intermediate_layer_model = Model(input=self.model.input,
				output=self.model.get_layer(layer_name).output)
			intermediate_output = intermediate_layer_model.predict(self.validation_data[0])

			#print (intermediate_layer_model.summary())

			get_layer_output = K.function([self.model.layers[0].input ],
										  [self.model.layers[0].output])
			layer_output = get_layer_output([self.validation_data[0]])[0]

			print (self.validation_data[0].dtype)
			print (intermediate_output.dtype)
			print (layer_output.dtype)
			curr_epoch = len(self.losses); 
			for slId in range(5):
				fn = os.path.join(self.path_out,"{:03d}_{:03d}_Input.mha".format(
					curr_epoch, slId))
				vol = sitk.GetImageFromArray(self.validation_data[0][slId,:,:,:])
				#cv2.imwrite(fn, self.validation_data[0][slId,:,:,:])	
				sitk.WriteImage(vol,fn)

				for kernelId in range(intermediate_output.shape[3]):
					sl = intermediate_output[slId,:,:,kernelId]*255
					print(kernelId, np.min(sl), np.max(sl))
					fn = os.path.join(self.path_out,
						"{:03d}_{:03d}_Kernel{:03d}.mha".format(curr_epoch, slId, kernelId))
					vol = sitk.GetImageFromArray(sl)
					sitk.WriteImage(vol,fn)

				fn = os.path.join(self.path_out,"{:03d}_{:03d}_Output.mha".format(curr_epoch,
					slId))
				vol = sitk.GetImageFromArray(np.abs(layer_output[slId,:,:,:]-
					self.validation_data[0][slId,:,:,:]))	
				sitk.WriteImage(vol,fn)

		except Exception as e:
			print e

		#write layers
		if False:
			for idxLayer, layer in enumerate(self.model.layers):
				weights = layer.get_weights() # list of numpy arrays
				print idxLayer, layer.__class__.__name__,
				if len(weights)>0: 
					try:
						print(K.min(weights), K.max(weights))	
					except Exception as e:
						print e
				else:
					print

def train_and_predict(data_path, img_rows, img_cols, n_epoch, input_no  = 3, output_no = 3,
	fn= "model"):
	
	print('-'*30)
	print('Loading and preprocessing train data...')
	print('-'*30)
	imgs_train, msks_train = load_data(data_path,"_train")
	imgs_train, msks_train = update_channels(imgs_train, msks_train, input_no, output_no)
	imgs_train /= 255.0
	#mean = np.mean(imgs_train)	# mean for data centering
	#std  = np.std(imgs_train)  # std for data normalization
	#imgs_train -= mean
	#imgs_train /= std
	
	print('-'*30)
	print('Loading and preprocessing test data...')
	print('-'*30)
	imgs_test, msks_test = load_data(data_path,"_test")
	imgs_test, msks_test = update_channels(imgs_test, msks_test, input_no, output_no)

	imgs_test /= 255.0


	print('-'*30)
	print('Creating and compiling model...')
	print('-'*30)
	model		= model5_MultiLayer(False, False, img_rows, img_cols, input_no, 
		output_no)
	model_fn	= os.path.join(data_path, fn+'_{epoch:03d}.hdf5')
	print ("Writing model to ", model_fn)
	
	#
	#various callback functions
	#
	model_checkpoint = ModelCheckpoint(model_fn, 
		monitor='loss', 
		save_best_only=False) # saves all models when set to False
	
	model_earlyStop = EarlyStopping(monitor='val_loss',	
		min_delta=0, 
		patience=0, 
		verbose=0, 
		mode='auto')
	
	model_customCallback = customCallback(validation_data=(imgs_test, msks_test),
		path_out = data_path )

	print('-'*30)
	print('Fitting model...')
	print('-'*30)
	history = History()
	history = model.fit(imgs_train, msks_train, 
		batch_size=128, 
		nb_epoch=n_epoch, 
		validation_data = (imgs_test, msks_test),
		verbose=1, 
		callbacks=[model_checkpoint, model_customCallback])

	print(history.history)

	json_fn = os.path.join(data_path, fn+'.json')
	with open(json_fn,'w') as f:
		f.write(model.to_json())


	print('-'*30)
	print('Loading saved weights...')
	print('-'*30)
	epochNo = len(history.history['loss'])-1
	model_fn	= os.path.join(data_path, '{}_{:03d}.hdf5'.format(fn, epochNo))
	model.load_weights(model_fn)

	print('-'*30)
	print('Predicting masks on test data...')
	print('-'*30)
	msks_pred = model.predict(imgs_test, verbose=1)
	print("Done ", epochNo, np.min(msks_pred), np.max(msks_pred))
	np.save(os.path.join(data_path, 'msks_pred.npy'), msks_pred)

	scores = model.evaluate(imgs_test,
		msks_test,
		batch_size=128, 
		verbose = 2)
	print ("Evaluation Scores", scores)

if __name__ =="__main__":
	train_and_predict(settings.OUT_PATH, settings.IMG_ROWS, settings.IMG_COLS, \
		settings.EPOCHS, settings.IN_CHANNEL_NO/settings.RESCALE_FACTOR, \
		settings.OUT_CHANNEL_NO/settings.RESCALE_FACTOR, settings.MODEL_FN)


