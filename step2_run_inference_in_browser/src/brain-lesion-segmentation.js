// Start timing now
console.time("all");


var ndarray = require("ndarray")
var ops = require("ndarray-ops")

//getTensor from canvas
function getTensor(canvasID)
{
    const inCtx = document.getElementById(canvasID).getContext('2d');
    const img = inCtx.getImageData( 0, 0,
      inCtx.canvas.width,
      inCtx.canvas.height
    );
    const { data, width, height } = img;
  
    let dataTensor = ndarray(new Float32Array(data), [width, height, 4 ]);
  
    return dataTensor;
}

//add buffer to canvas
function addToCanvas(canvasID, buffer, width, height)
{
  //get the canvas
  const outCtx = document.getElementById(canvasID).getContext('2d');

  // create imageData object
  var idata = outCtx.createImageData(width, height);

  // set our buffer as source
  idata.data.set(buffer);

  // update canvas with new data
  outCtx.putImageData(idata, 0, 0);
}

const model = new KerasJS.Model({
  filepaths: {
    model: '../demos/data/brain_habitat_seg_128/brainHabitats.json',
    weights: '../demos/data/brain_habitat_seg_128/brainHabitats_weights.buf',
    metadata: '../demos/data/brain_habitat_seg_128/brainHabitats_metadata.json'
  },
  gpu: true,
  dim_ordering: 'tf'
});

model.ready()
  .then(() => {

    width = 128;
    height = 128;
    let dataTensorT1 = getTensor('t1');
    let dataTensorT2 = getTensor('t2');
    let dataTensorFl = getTensor('fl');
 
    let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      width,
      height,
      3
    ]);
  
    ops.assign(
      dataProcessedTensor.pick(null, null, 0),
      dataTensorT1.pick(null, null, 0)
    );
  
    ops.assign(
      dataProcessedTensor.pick(null, null, 1),
      dataTensorT2.pick(null, null, 0)
    );
  
    ops.assign(
      dataProcessedTensor.pick(null, null, 2),
      dataTensorFl.pick(null, null, 0)
    );
   
    const inputData = { 'input_1': dataProcessedTensor.data }

    // make predictions
    model.predict(inputData)
      .then(outputData => {
        this.output = outputData['convolution2d_19'];
      
        var width   = 128;
        var height  = 128;
        var bufferEnhCore  = new Uint8ClampedArray(width*height*4)
        var bufferNecrosis = new Uint8ClampedArray(width*height*4)
        var bufferEdema    = new Uint8ClampedArray(width*height*4)
        
        for(var y = 0; y < height; y++) {
          for(var x = 0; x < width; x++) {
            var pos = (y * width + x) * 4; // position in buffer based on x and y
            if (this.output[0+(x+y*width)*3]>0.9)
              {
                bufferEnhCore[pos  ] = dataProcessedTensor.data[0+(x+y*width)*3];
                bufferEnhCore[pos+1] = 0;
                bufferEnhCore[pos+3] = 0;
              }
            else
              {
                bufferEnhCore[pos  ] = dataProcessedTensor.data[0+(x+y*width)*3];
                bufferEnhCore[pos+1] = dataProcessedTensor.data[0+(x+y*width)*3];
                bufferEnhCore[pos+2] = dataProcessedTensor.data[0+(x+y*width)*3];
              }

             if (this.output[1+(x+y*width)*3]>0.9)
              {

                bufferNecrosis[pos  ] = dataProcessedTensor.data[1+(x+y*width)*3];
                bufferNecrosis[pos+1] = 0;
                bufferNecrosis[pos+2] = 0;
              }
            else
              {
                bufferNecrosis[pos  ] = dataProcessedTensor.data[1+(x+y*width)*3];
                bufferNecrosis[pos+1] = dataProcessedTensor.data[1+(x+y*width)*3];
                bufferNecrosis[pos+2] = dataProcessedTensor.data[1+(x+y*width)*3];
              }
              
             if (this.output[2+(x+y*width)*3]>0.9)
              {
                bufferEdema[pos  ] = dataProcessedTensor.data[2+(x+y*width)*3];
                bufferEdema[pos+1] = 0;
                bufferEdema[pos+2] = 0;
              }
              else
              {
                bufferEdema[pos  ] = dataProcessedTensor.data[2+(x+y*width)*3];
                bufferEdema[pos+1] = dataProcessedTensor.data[2+(x+y*width)*3];
                bufferEdema[pos+2] = dataProcessedTensor.data[2+(x+y*width)*3];
              }
            
            
            bufferEnhCore[pos+3] = 255;           
            bufferNecrosis[pos+3] = 255;        
            bufferEdema[pos+3] = 255;           
            
          }
        }
      
        addToCanvas('EnhCore', bufferEnhCore, width, height);
        addToCanvas('Necrosis', bufferNecrosis, width, height);
        addToCanvas('Edema', bufferEdema, width, height);
        document.getElementById('status').innerHTML = 'Done';
        return null; 
      })
      .catch(err => {
        console.log(err);
        console.log('Error predicting');
      })
    return null; 
  } )
  .catch(err => {
     console.log(err);
  });