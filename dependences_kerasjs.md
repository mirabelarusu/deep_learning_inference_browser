# Dependences [Keras.js](https://github.com/transcranial/keras-js)

This document summaries dependences to be able to use keras.js and run inference in the browser. 

The current version of this instructions does not include instruction for vue, [node.js](http://blog.teamtreehouse.com/install-node-js-npm-windows)

Install webpack (Instructions from [here](http://webpack.github.io/docs/tutorials/getting-started/) and [here](https://webpack.js.org/guides/get-started/?_sm_au_=i5H1Wsj42j1BHnDV))

```npm install webpack -g```


First install

```npm install css-loader style-loader```

Needed by keras.js

```sh
npm install babel-polyfill
npm install babel-loader
npm install babel-core
npm i babel-plugin-transform-class-properties
npm i babel-plugin-transform-object-rest-spread
npm i babel-plugin-transform-async-to-module-method
npm install --save-dev babel-cli babel-preset-latest

npm install precss
npm install  postcss-loader
npm install axios
```

for the convolutional network

```sh
npm install ndarray-unsqueeze
npm install ndarray-tile
npm install ndarray-concat-rows
npm install ndarray-gemm

npm install ndarray-unpack
npm install ndarray-squeeze
npm install ndarray

npm install bluebird
npm install weblas

npm install ndarray-blas-level2
npm install ndarray-ops
npm install cwise
npm install glslify-loader
npm install raw-loader
```

Get the http server 
```sh
npm install -g http-server
```

```sh
npm install ndarray
npm install ndarray-ops
npm install -g browserify
```


