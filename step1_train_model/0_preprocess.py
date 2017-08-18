""" To conver dicom images into images needed for keras"""

from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
import settings
import cv2
import random
def harmonizeToNormal(sitkImgIn, sitkMask, normalLabel = 10):

	imgArray = sitk.GetArrayFromImage(sitkImgIn)
	mskArray = sitk.GetArrayFromImage(sitkMask)

	avg = np.average(imgArray[mskArray == normalLabel])

	if (avg!=0): # needed
		sitkImgOut = sitk.Cast(sitkImgIn,sitk.sitkFloat32) 
		sitkImgOut = sitk.Multiply (sitkImgOut, 1.0/avg)
	else:
	    print("Couldn't standardize due to division by zero")
	    sitkImgOut = sitkImgIn

	return sitkImgOut

def apply_bias_field_correction(img, msk, numControlPoints=8, shrinkFactor=4, numFittingLevels=3, numIterationAtEachLevel=5, FWHM=0.15):
	"""Runs N4 bias-correction on img (nibabel image) with mask msk  (nibabel image).
     Returns bias-corrected image (nibabel image) and  bias-field (SimpleITK image)
	"""

	print("Doing Bias Field Correction")
	fImg  = sitk.Cast(img,sitk.sitkFloat32)
	iMsk = sitk.Cast(msk, sitk.sitkInt32 )

	fImg = sitk.DiscreteGaussian(fImg,3)


	corrector = sitk.N4BiasFieldCorrectionImageFilter();

	#per 3D Slicer suggestion
	corrector.SetNumberOfControlPoints([numControlPoints,numControlPoints,
		numControlPoints]);
	corrector.SetMaximumNumberOfIterations([int(numIterationAtEachLevel)]*numFittingLevels)
	corrector.SetConvergenceThreshold(0.0001)
	corrector.SetBiasFieldFullWidthAtHalfMaximum(FWHM)
	corrector.SetSplineOrder(3)

	BSImg = sitk.Shrink( fImg, [shrinkFactor,shrinkFactor,shrinkFactor]*img.GetDimension())
	BSMsk = sitk.Shrink( iMsk, [shrinkFactor,shrinkFactor,shrinkFactor]*msk.GetDimension())

	BSImgCor = corrector.Execute(BSImg, BSMsk) 


	bfieldS = sitk.Divide(BSImg, BSImgCor) # Not a real bias-field outside mask!
	bfieldS = sitk.Resample(bfieldS, fImg, sitk.Transform(), sitk.sitkLinear, 1.0, \
		sitk.sitkUnknown)

	BSImgCor = sitk.Divide(sitk.Cast(img,sitk.sitkFloat32), 
		sitk.Cast(bfieldS,sitk.sitkFloat32))

	return BSImgCor

def get_data_from_dir(data_dir):
	"""
	From a given folder (in the Brats2016 folder organization), returns the different 
	volumes corresponding to t1, t1c, f 
	"""
	print ("Loading from", data_dir)
	img_path	= os.path.dirname(data_dir)
	img_dir_fn	= os.path.basename(data_dir)
	t1_fn		= ""
	t1c_fn		= ""
	flair_fn	= ""
	t2_fn		= ""
	truth_fn	= ""

	fldr1_list = os.listdir(data_dir)
	for fldr1 in fldr1_list:
		fldr1_fn = os.path.join(img_path,img_dir_fn, fldr1)
		if os.path.isdir(fldr1_fn): 
			fldr2_list = os.listdir(fldr1_fn)
			for fldr2 in fldr2_list:
				fn, ext = os.path.splitext(fldr2)
				if ext == '.mha':
					protocol_series = fldr1.split('.')[4]
					protocol = protocol_series.split('_')[0]
					if protocol == 'MR':
						series = protocol_series.split('_')[1]
						if series == 'T2':
							t2_fn = os.path.join(img_path,img_dir_fn, fldr1, fldr2)
						if series == 'Flair':
							flair_fn = os.path.join(img_path, img_dir_fn, fldr1, fldr2)
						if series == 'T1c':
							t1c_fn = os.path.join(img_path,img_dir_fn, fldr1, fldr2)
						if series == 'T1':
							t1_fn = os.path.join(img_path,img_dir_fn, fldr1, fldr2)
					else:
						truth_fn = os.path.join(img_path,img_dir_fn, fldr1, fldr2)

	#does the data have all the needed inputs: T1C, T2, Flair and truth, them use
	isComplete = False
	if len(t1c_fn)>0 and len(t1_fn) and len(flair_fn)>0 and len(t2_fn)>0 \
		and len(truth_fn)>0:
		isComplete = True
		print ("  T1 :", os.path.basename(t1_fn))
		print ("  T1c:", os.path.basename(t1c_fn)) 
		print ("  FLr:", os.path.basename(flair_fn)) 
		print ("  T2 :", os.path.basename(t2_fn))
		print ("  Tru:", os.path.basename(truth_fn))

	# Read data
	try:
		t1 = sitk.ReadImage(t1_fn)
	except Exception as e:
		print (e)
		t1 = sitk.Image()
	
	try:
		t1c = sitk.ReadImage(t1c_fn)
	except Exception as e:
		print (e)
		t1c = sitk.Image()
	
	try:
		fl = sitk.ReadImage(flair_fn)
	except Exception as e:
		print (e)
		fl = sitk.Image()

	try:
		t2 = sitk.ReadImage(t2_fn)
	except Exception as e:
		print (e)
		t2 = sitk.Image()
	
	try:
		msk = sitk.ReadImage(truth_fn);
		msk.SetOrigin(t1.GetOrigin())
		msk.SetDirection(t1.GetDirection())
		msk.SetSpacing(t1.GetSpacing())
	except Exception as e:
		print (e)
		msk = sitk.Image()

	return (t1, t1c, fl, t2, msk, isComplete);

def preprocessSITK(img, img_rows, img_cols, resize_factor=1, 
	applyBiasFieldCorrection = 0):
	"""
		crops, rescales, does the bias field correction on an sitk image
	----
	Input: sitk image
	Output: sitk image
	"""
	si_img = img.GetSize()
	sp_img = img.GetSpacing()
	
	#crop to the desired size:
	low_boundary	= [int((si_img[0]-img_rows)/2),int((si_img[1]-img_cols)/2), 0]
	upper_boundary	= [int((si_img[0]-img_rows+1)/2),int((si_img[1]-img_cols+1)/2),0]
	
	pr_img = sitk.Crop(img, low_boundary, upper_boundary)

	### apply bias field correction
	if applyBiasFieldCorrection:
		pr_img = apply_bias_field_correction(pr_img, sitk.Cast(pr_img>0, sitk.sitkInt8))

	if not resize_factor==1:
		pr_img = sitk.Shrink(pr_img,[resize_factor, resize_factor, 1])
		print ("Resizing to", pr_img.GetSize())

	return pr_img

def preprocessNP(img_arr, tumor_arr = None):
	"""
	intensity preprocessing: normaling to normal region, rescale to 0-255 range
	"""
	normal_arr = (img_arr>0).astype('uint8')

	if tumor_arr is not None:
		normal_arr[tumor_arr>0] = 0

	img_arr /= np.mean(img_arr[normal_arr>0])
	new_img_arr = img_arr/float(np.max(img_arr))*255.

	return new_img_arr, normal_arr

def create_datasets_4(img_path, img_rows, img_cols, img_slices, slice_by=5, resize_factor = 1, out_path='.'):
	"""
	creates training with 4 Inputs, and 5 outputs (1-necrosis,2-edema, 
	3-non-enhancing-tumor, 4-enhancing tumore, 5 - rest brain)
	"""

	img_list = os.listdir(img_path)

	slices_per_case = 155
	n_labels = 5
	n_inputs = 4

	img_rows_ss = img_rows/resize_factor
	img_cols_ss = img_cols/resize_factor

	#training
	tr_n_cases = 273 # tcia
	tr_n_slices = slices_per_case*tr_n_cases
	tr_label_counts = np.zeros(n_labels+2)

	tr_img_shape = (tr_n_slices, img_rows_ss, img_cols_ss, n_inputs)
	tr_msk_shape = (tr_n_slices, img_rows_ss, img_cols_ss, n_labels)

	tr_imgs = np.ndarray(tr_img_shape, dtype=np.uint16)
	tr_msks = np.ndarray(tr_msk_shape, dtype=np.uint16)
	# slices from the train set but not used for training
	ntr_imgs = np.ndarray(tr_img_shape, dtype=np.uint16)
	ntr_msks = np.ndarray(tr_msk_shape, dtype=np.uint16)


	#testing
	te_n_cases = 60 
	te_n_slices = slices_per_case*te_n_cases
	te_img_shape = (te_n_slices, img_rows_ss, img_cols_ss, n_inputs)
	te_msk_shape = (te_n_slices, img_rows_ss, img_cols_ss, n_labels)

	te_imgs = np.ndarray(te_img_shape, dtype=np.uint16)
	te_msks = np.ndarray(te_msk_shape, dtype=np.uint16)

	#a subset that is selected for the user experiments
	te_usr_imgs = np.ndarray(te_img_shape, dtype=np.uint16)
	te_usr_msks = np.ndarray(te_msk_shape, dtype=np.uint16)
	#slices from the testing set, not used for testing
	nte_imgs = np.ndarray(te_img_shape, dtype=np.uint16)
	nte_msks = np.ndarray(te_msk_shape, dtype=np.uint16)

	te_label_counts = np.zeros(n_labels+2)
	te_img_shape_3D = (te_n_cases, slices_per_case, img_rows_ss, img_cols_ss, n_inputs)
	te_msk_shape_3D = (te_n_cases, slices_per_case, img_rows_ss, img_cols_ss, n_labels)

	#automatic testing
	te_au_imgs_3D = np.ndarray(te_img_shape_3D, dtype=np.uint16)
	te_au_msks_3D = np.ndarray(te_msk_shape_3D, dtype=np.uint16)

	#user testing
	te_us_imgs_3D = np.ndarray(te_img_shape_3D, dtype=np.uint16)
	te_us_msks_3D = np.ndarray(te_msk_shape_3D, dtype=np.uint16)

	i = 0
	print('-'*30)
	print('Creating training images...')
	print('-'*30)
	tr_i		= 0
	te_i		= 0

	slicesTr	= 0
	slicesTe	= 0
	curr_sl_tr	= 0
	curr_sl_te	= 0
	curr_sl_usr_te = 0
	curr_cs_te	= 0
	curr_sl_ntr	= 0
	curr_sl_nte	= 0
	curr_cs_usr_te = 0


	n_truth		= 0
	n_total		= 0

	for i, img_dir_fn in enumerate(img_list):
		data_dir = os.path.join(img_path,img_dir_fn)
		# skip if is not a folder
		if not os.path.isdir(data_dir):
			continue

		# find out which on is in training 
		is_tr = True;
		#if img_dir_fn.split('_')[1] == "2013":
		if i % 5 == 0:
			is_tr = False
			if i % 10 == 0:
				is_au = True
			else:
				is_au = False


		print (i, "Train:", is_tr, "", end='')
		(t1p, t1, fl, t2, msk, isComplete) = get_data_from_dir(data_dir)

		#preprocess
		applyBS = True
		t1	= preprocessSITK(t1,img_rows, img_cols, resize_factor, applyBS)
		t1p	= preprocessSITK(t1p,img_rows, img_cols, resize_factor, applyBS)
		fl	= preprocessSITK(fl,img_rows, img_cols, resize_factor, applyBS)
		t2	= preprocessSITK(t2,img_rows, img_cols, resize_factor, applyBS)
		msk	= preprocessSITK(msk,img_rows, img_cols, resize_factor, False)

		imgArr = np.zeros((slices_per_case, img_rows_ss, img_cols_ss,n_inputs))	
		imgArr[:,:,:,0]	= sitk.GetArrayFromImage(t1).astype('float')
		imgArr[:,:,:,1]	= sitk.GetArrayFromImage(t2).astype('float')
		imgArr[:,:,:,2]	= sitk.GetArrayFromImage(fl).astype('float')
		imgArr[:,:,:,3]	= sitk.GetArrayFromImage(t1p).astype('float')

	
		mskArr = np.zeros((slices_per_case, img_rows_ss, img_cols_ss,n_labels))
		mskArrTmp = sitk.GetArrayFromImage(msk)
		mskArr[:,:,:,0] = (mskArrTmp==1).astype('float')
		mskArr[:,:,:,1] = (mskArrTmp==2).astype('float')
		mskArr[:,:,:,2] = (mskArrTmp==3).astype('float')
		mskArr[:,:,:,3] = (mskArrTmp==4).astype('float')

		#normalizes intensities, and creates normal roi region
		normalRoiArr = np.zeros((slices_per_case, img_rows_ss, img_cols_ss))
		for idxIn in range(n_inputs):
			imgArr[:,:,:,idxIn], norTmp = preprocessNP(imgArr[:,:,:,idxIn], mskArrTmp)

			#add nornal 
			normalRoiArr[norTmp>0] = 1

		#remove tumor
		normalRoiArr[mskArrTmp>0] = 1
		mskArr[:,:,:,4] = normalRoiArr

		n_slice = 0
		minSlice = 0
		maxSlice = slices_per_case
		for curr_slice in range(slices_per_case):#leasionSlices:
			n_slice +=1
			# is slice in training cases, but not used from training,or testin 
			#in the first state
			if n_slice % slice_by == 0:
				print ('.', sep='', end='')
				is_used = True
			else:
				is_used = False

			imgSl = imgArr[curr_slice,:,:,:]
			mskSl = mskArr[curr_slice,:,:,:]

			# set slice
			if is_tr:
				for l in range(1,n_labels):
					tr_label_counts[l] += len(np.where(mskSl[:,:,l-1] == 1)[0])
					
				tr_label_counts[n_labels] += len(np.where(mskSl[:,:,n_labels-1]>0)[0])
				tr_label_counts[n_labels+1] += mskArr.shape[1]*mskArr.shape[2]

				# regular training slices
				if is_used:
					if curr_sl_tr % 2 == 0:
						tr_imgs[curr_sl_tr,:,:,:] = imgSl
						tr_msks[curr_sl_tr,:,:,:] = mskSl
					else: 
						tr_imgs[curr_sl_tr,:,:,:] = cv2.flip(imgSl,1)
						tr_msks[curr_sl_tr,:,:,:] = cv2.flip(mskSl,1)
					curr_sl_tr += 1
				else: # not used in the first step of training
					ntr_imgs[curr_sl_ntr,:,:,:] = imgSl
					ntr_msks[curr_sl_ntr,:,:,:] = mskSl
					curr_sl_ntr += 1


				n_truth += len(np.where(mskSl>0)[0])
				n_total += mskArr.shape[1]*mskArr.shape[2]
			else:
				for l in range(1,n_labels):
					te_label_counts[l] += len(np.where(mskSl[:,:,l-1] == 1)[0])
				te_label_counts[n_labels] += len(np.where(mskSl[:,:,n_labels-1]>0)[0])
				te_label_counts[n_labels+1] += mskArr.shape[1]*mskArr.shape[2]
	
				if is_au: # is part of the automatic testing
					if is_used:
						te_imgs[curr_sl_te,:,:,:] = imgSl
						te_msks[curr_sl_te,:,:,:] = mskSl
						curr_sl_te += 1
					else:
						nte_imgs[curr_sl_nte,:,:,:] = imgSl
						nte_msks[curr_sl_nte,:,:,:] = mskSl
						curr_sl_nte += 1
				else: # part of the user test group
					if is_used: 
						te_usr_imgs[curr_sl_usr_te,:,:,:] = imgSl
						te_usr_msks[curr_sl_usr_te,:,:,:] = mskSl
						curr_sl_usr_te += 1
	
				
		#new line needed for the ... simple progress bar
		print ('\n')
	

		if is_tr:
			tr_i += 1
			slicesTr += maxSlice - minSlice+1 
		else:
			te_i += 1
			slicesTe += maxSlice - minSlice+1
			if is_au:
				te_au_imgs_3D[curr_cs_te,:,:,:,:] = imgArr
				te_au_msks_3D[curr_cs_te,:,:,:,:] = mskArr
				curr_cs_te +=1
			else:
				te_us_imgs_3D[curr_cs_usr_te,:,:,:,:] = imgArr
				te_us_msks_3D[curr_cs_usr_te,:,:,:,:] = mskArr
				curr_cs_usr_te +=1
			
		


		if (tr_i+te_i) % 10 == 0:
			print('Done: {0}/{1} images, {2} {3}'.format(tr_i+te_i, 
				tr_n_cases+ te_n_cases, curr_sl_tr, curr_sl_te))
			for l in range(1,n_labels+1):
				print("tr -{:02d} {:8.6f}".format(l, 
					tr_label_counts[l]/float(tr_label_counts[n_labels])))
				print("te -{:02d} {:8.6f}".format(l, 
					te_label_counts[l]/float(te_label_counts[n_labels])))

	print('Done loading .',slicesTr, slicesTe, curr_sl_tr, curr_sl_te, n_truth, 
		n_total, n_truth/float(n_total+1e-6) )
	for l in range(1,n_labels+1):
		print("tr -{:02d} {:8.6f}".format(l, 
			tr_label_counts[l]/float(tr_label_counts[n_labels])))
		print ("te -{:02d} {:8.6f}".format(l, 
			te_label_counts[l]/float(te_label_counts[n_labels])))


	tr_imgs = tr_imgs[0:curr_sl_tr,:,:,:]
	tr_msks = tr_msks[0:curr_sl_tr,:,:,:]

	np.save(os.path.join(out_path,'imgs_train.npy'), tr_imgs)
	np.save(os.path.join(out_path,'msks_train.npy'), tr_msks)

	te_imgs = te_imgs[0:curr_sl_te,:,:,:]
	te_msks = te_msks[0:curr_sl_te,:,:,:]

	np.save(os.path.join(out_path,'imgs_test.npy'),  te_imgs)
	np.save(os.path.join(out_path,'msks_test.npy'),  te_msks)
	
	te_usr_imgs = te_usr_imgs[0:curr_sl_usr_te,:,:,:]
	te_usr_msks = te_usr_msks[0:curr_sl_usr_te,:,:,:]

	np.save(os.path.join(out_path,'msks_test_usr.npy'),  te_usr_msks)
	np.save(os.path.join(out_path,'imgs_test_usr.npy'),  te_usr_imgs)

	ntr_imgs = ntr_imgs[0:curr_sl_ntr,:,:,:]
	ntr_msks = ntr_msks[0:curr_sl_ntr,:,:,:]
	np.save(os.path.join(out_path,'imgs_not_used_train.npy'), ntr_imgs)
	np.save(os.path.join(out_path,'msks_not_used_train.npy'), ntr_msks)

	nte_imgs = nte_imgs[0:curr_sl_nte,:,:,:]
	nte_msks = nte_msks[0:curr_sl_nte,:,:,:]

	np.save(os.path.join(out_path,'imgs_not_used_test.npy'),  nte_imgs)
	np.save(os.path.join(out_path,'msks_not_used_test.npy'),  nte_msks)

	te_au_imgs_3D = te_au_imgs_3D[0:curr_cs_te,:,:,:,:]
	te_au_msks_3D = te_au_msks_3D[0:curr_cs_te,:,:,:,:]

	np.save(os.path.join(out_path,'imgs_test_3D.npy'),  te_au_imgs_3D)
	np.save(os.path.join(out_path,'msks_test_3D.npy'),  te_au_msks_3D)


	te_us_imgs_3D = te_us_imgs_3D[0:curr_cs_usr_te,:,:,:,:]
	te_us_msks_3D = te_us_msks_3D[0:curr_cs_usr_te,:,:,:,:]
	
	np.save(os.path.join(out_path,'imgs_test_3D_usr.npy'),  te_us_imgs_3D)
	np.save(os.path.join(out_path,'msks_test_3D_usr.npy'),  te_us_msks_3D)
	
	print('Saving to .npy files done.')
	print('Train   : ', curr_sl_tr)
	print('Test    : ', curr_sl_te)
	print('Test urs: ', curr_sl_usr_te)
	print('Test 3D : ', curr_cs_te)
	print('Test3urs: ', curr_cs_usr_te)

def load_data(data_path, prefix = "_train"):
	imgs_train = np.load(os.path.join(data_path, 'imgs'+prefix+'.npy'))
	msks_train = np.load(os.path.join(data_path, 'msks'+prefix+'.npy'))
	return imgs_train, msks_train

def update_channels(imgs, msks, input_no=3, output_no=3, mode=1):
	"""
	changes the order or which channels are used to allow full testing. Uses both
	Imgs and msks as input since different things may be done to both
	---
	mode: int between 1-3

	"""
	imgs = imgs.astype('float32')
	msks = msks.astype('float32')

	shp = imgs.shape
	new_imgs = np.zeros((shp[0],shp[1],shp[2],input_no))
	new_msks = np.zeros((shp[0],shp[1],shp[2],output_no))

	if mode==1:
		new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]
	elif mode == 2:
		new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,2]+msks[:,:,:,3]
		print('-'*10,' Predicing active Core', '-'*10)
	elif mode == 3:
		#core (non enhancing)
		new_msks[:,:,:,0] = msks[:,:,:,3]
		#print('-'*10,' Predicing enhancing tumor', '-'*10)
	elif mode == 4:
		#randombly select slices and randomly choose which of the lables to add
		#simple way to simultate partial labels.i
		print(new_msks.shape)
		for sl in range(new_mask.shape[0]):
			for idxOut in range(output_no):
				ran = round(random.random())
				print(idxOut, ran)
				if idxOut==0:
					new_msks[sl,:,:,0] = ran*msks[:,:,:,idxOut]
				else:
					new_msks[sl,:,:,0] += ran*msks[:,:,:,idxOut]

	else:
		new_msks[:,:,:,0] = msks[:,:,:,0]+msks[:,:,:,1]+msks[:,:,:,2]+msks[:,:,:,3]

	if input_no == 1: 
		if mode==1:
			new_imgs[:,:,:,0] = imgs[:,:,:,2]
		elif mode==2:
			new_imgs[:,:,:,0] = imgs[:,:,:,0]
		else:
			new_imgs[:,:,:,0] = imgs[:,:,:,0]
	
	if input_no  == 2: 
		new_imgs[:,:,:,:] = imgs[:,:,:,0:2]
	
	if output_no == 2:
		new_msks[:,:,:,1] = msks[:,:,:,3]

	if input_no == 3: 
		print (new_imgs[:,:,:,0].shape, imgs[:,:,:,1].shape)
		new_imgs[:,:,:,0] = imgs[:,:,:,1]
		new_imgs[:,:,:,1] = imgs[:,:,:,0]
		new_imgs[:,:,:,2] = imgs[:,:,:,2]
	
	if output_no == 3:
		new_msks[:,:,:,1] = new_msks[:,:,:,0]
		new_msks[:,:,:,2] = new_msks[:,:,:,0]

	return new_imgs, new_msks

if __name__ == '__main__':
	
	data_path = settings.DATA_PATH
	out_path  = settings.OUT_PATH

	img_rows  = settings.IMG_ROWS
	img_cols  = settings.IMG_COLS
	img_slices= 1;

	"1 - consider all slices"
	"5 - consider very firth slices - for time purposes"
	slice_by   = settings.SLICE_BY 

	rescale_factor = settings.RESCALE_FACTOR

	create_datasets_4(data_path, img_rows,img_cols, img_slices, slice_by, rescale_factor, 
		out_path)

