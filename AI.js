let genderAI;


$(document).ready(function() {
  alert('Please Allow Webcam Access.'+ 
    ' Website is not tracking or gathering information.' + 
    ' model runs locally on Machine and accessing Website through Mobile is not adviced.');
  run();
})


Array.prototype.reshape = function(rows, cols) {
  var copy = this.slice(0); 
  this.length = 0; 

  for (var r = 0; r < rows; r++) {
    var row = [];
    for (var c = 0; c < cols; c++) {
      var i = r * cols + c;
      if (i < copy.length) {
        row.push(copy[i]);
      }
    }
    this.push(row);
  }
};


function drawconv_map(x,elements_name,reshape_x,reshape_y,width_plot,height_plot)
{
  
  x.reshape(reshape_x,reshape_y);
  var data = [
  {
    z: x,
    type: 'heatmap',
    colorscale: 'YIGnBu',
    showlegend: false,
    showarrow: false,
    showscale: false,
    showgrid : false
  }
  ];

  var layout = {
    autosize: false,
    width: width_plot,
    height: height_plot,
    margin: {
      l: 10,
      r: 10,
      b: 10,
      t: 10,
      pad: 4
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    xaxis: {visible: false},
    yaxis: {visible: false},

  };

  Plotly.newPlot(elements_name, data , layout);
}



async function run() {

  var video = document.getElementById('video');
  var canvas = document.getElementById('canvas');
  var context = canvas.getContext('2d');
  var prediction_display = document.getElementById('prediction')


  genderAI = await tf.loadModel('res/model/model.json');
  
  var tracker = await new tracking.ObjectTracker('face');

  tracker.setInitialScale(4);
  tracker.setStepSize(2);
  tracker.setEdgesDensity(0.2);
  tracking.track('#webcam', tracker, { camera: true });
  
  tracker.on('track', function(event) 
  {
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    event.data.forEach(function(rect)
    {
      context.strokeStyle = '#a64ceb';
      context.strokeRect(rect.x, rect.y, rect.width, rect.height);
      context.font = '11px Helvetica';
      context.fillStyle = "#fff";
      context.fillText('x: ' + rect.x + 'px', rect.x + rect.width + 5, rect.y + 11);
      context.fillText('y: ' + rect.y + 'px', rect.x + rect.width + 5, rect.y + 22);

      
      if(rect.x + rect.width < 360 && rect.y + rect.height < 320)
      { 

       const webcamImage = tf.fromPixels(document.getElementById('webcam'));
       const croppedImage = webcamImage.slice([rect.y, rect.x, 0], [rect.width + 5, rect.height + 11 , 3]);
       
       
       const feedImage = tf.image.resizeBilinear(croppedImage,[50,50]);
       
       var input = []
       input.push(tf.tidy(() => { return tf.expandDims(feedImage,0).asType('float32').div(255.0)}));
       
       
       for (var i = 1; i <= 12; i++) {
         input.push(genderAI.layers[i].apply(input[i-1]));
       }
       
       
       const firstconv = input[2];
       const secondconv = input[4];
       const thirdconv = input[6];
       const firstconv_list = tf.tidy(() => { return tf.unstack(firstconv.reshape([25,25,16]),2)});
       const secondconv_list = tf.tidy(() => { return tf.unstack(secondconv.reshape([12,12,32]),2)});
       const thirdconv_list = tf.tidy(() => { return tf.unstack(thirdconv.reshape([6,6,32]),2)});
       console.log(firstconv_list[2].print())
       for(var i = 0 ; i < 16 ; i++)
       {
        const reverse_img = tf.reverse2d(firstconv_list[i]);
        drawconv_map(Array.from(reverse_img.dataSync()),"fc_"+i,25,25,150,150);
        reverse_img.dispose();
      }

      for(var i = 0 ; i < 32 ; i++)
      {
        const reverse_img = tf.reverse2d(secondconv_list[i]);
        const reverse_img_2 = tf.reverse2d(secondconv_list[i]);
        drawconv_map(Array.from(reverse_img.dataSync()),"sc_"+i,12,12,75,75);
        drawconv_map(Array.from(reverse_img.dataSync()),"tc_"+i,6,6,50,50);
        reverse_img_2.dispose();
        reverse_img.dispose();
      }                  
      const predData = input[12].dataSync();
      if (predData[0] > 0.5)
      {
        prediction_display.innerHTML = "Prediction : Male ( Confidence: " + Number(predData[0].toFixed(2)) + ")";
      }
      else
      {
        prediction_display.innerHTML = "Prediction : Female ( Confidence: " + Number((1 - predData[0]).toFixed(2)) + ")";
      }

      webcamImage.dispose();
      croppedImage.dispose();
      feedImage.dispose();
      firstconv.dispose();
      secondconv.dispose();
      thirdconv.dispose();

      for(var i = 1 ; i < 16 ; i++)
      {
        firstconv_list[i].dispose();
      }

      for (var i = 0; i < 32; i++) 
      {
        secondconv_list[i].dispose();
        thirdconv_list[i].dispose();
      }

      for (var i = input.length - 1; i >= 0; i--) {
        input[i].dispose();
      }
                  // console.log(tf.memory().numTensors);
                }
              });
  });

};
