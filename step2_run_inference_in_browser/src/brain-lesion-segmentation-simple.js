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
    model: '../../step1_train_model/Results/brainWholeTumor.json',
    weights: '../../step1_train_model/Results/brainWholeTumor_009_weights.buf',
    metadata: '../../step1_train_model/Results/brainWholeTumor_009_metadata.json'
  },
  gpu: true,
  dim_ordering: 'tf'
});

model.ready()
  .then(() => {

    width = 128;
    height = 128;
    let dataTensorFl = getTensor('fl');
 
    let dataProcessedTensor = ndarray(new Float32Array(width * height * 1), [
      width,
      height,
      1
    ]);
  
    ops.assign(
      dataProcessedTensor.pick(null, null, 0),
      dataTensorFl.pick(null, null, 0)
    );
   
    const inputData = { 'input_1': dataProcessedTensor.data }

    // make predictions
    model.predict(inputData)
      .then(outputData => {
        //name of the last layer
        this.output = outputData['convolution2d_19'];
      
        var width   = 128;
        var height  = 128;
        var bufferEdema    = new Uint8ClampedArray(width*height*4)
        
        for(var y = 0; y < height; y++) {
          for(var x = 0; x < width; x++) {
            var pos = (y * width + x) * 4; // position in buffer based on x and y
             if (this.output[x+y*width]>0.9)
             {
                bufferEdema[pos+0] = dataProcessedTensor.data[x+y*width];
                bufferEdema[pos+1] = 0;
                bufferEdema[pos+2] = 0;
              }
              else
              {
                bufferEdema[pos+0] = dataProcessedTensor.data[x+y*width];
                bufferEdema[pos+1] = dataProcessedTensor.data[x+y*width];
                bufferEdema[pos+2] = dataProcessedTensor.data[x+y*width];
              }
            
            bufferEdema[pos+3] = 255;           
            
          }
        }
      
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