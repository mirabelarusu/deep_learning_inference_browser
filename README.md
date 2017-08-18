# deep_learning_inference_browser
Example how to train a deep learning keras model to segment a tumor on brain MR and how to do deep learning inference in the browser


## Workflow

[Step 1: Train a keras model](step1_train_model)

[Step 2: Run inference in browser](step2_run_inference_in_browser)


## Code Requirements

Step 1: Train a keras model

1. [Keras](https://keras.io/) and all its dependences. [See installation instructions](https://keras.io/#installation)

Step 2: Run Inference in browser 

1. [encoder.py](https://github.com/transcranial/keras-js/blob/master/encoder.py) from [keras.js](https://github.com/transcranial/keras-js)

2. [keras.js](https://github.com/transcranial/keras-js) and its dependences. Instructions are provided [here](dependences_kerasjs.md) 

3. browserify to bundle the js files
```sh
npm install -g browserify
```

## Data Requirements

Step 1: [Brats data](https://sites.google.com/site/braintumorsegmentation/home/brats_2016) 2016 - segmentation of tumor subregions 

Step 2: Test images


## Usage

Step 1: Train model by running the preprocessing and training scripts: [0_preprocess.py](step1_train_model/0_preprocess.py) and [1_train.py](step1_train_model/1_train.py)


```sh
cd Step1_train_model

python 0_preprocess.py

python 1_train.py
```

The output of this step is a model (/path/to/model.hdf5) that needs to be encoded using [encoder.py](https://github.com/transcranial/keras-js/blob/master/encoder.py)

```sh
python encoder.py /path/to/model.hdf5
```

Step 2: Import in browser following the instructions 4-5 from [here](https://github.com/transcranial/keras-js#usage) steps 4-5

our code in [brain-lesion-segmentation.html](step2_run_inference_in_browser/src/brain-lesion-segmentation.html) starting line 14

```html
<script src="../ext/keras.js"></script>
```

and import our js from [here](step2_run_inference_in_browser/src/brain-lesion-segmentation.js)

```html
<script src="../dist/brain-habitat-segmentation-128.js"></script>
```


Uses browserify to create bundle js: 

```
browserify prostate-segmentation.js > ../dist/prostate-segmentation.js
```
