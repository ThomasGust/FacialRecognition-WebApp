$(document).ready(function(){
    //let namespace = "/v-stream";
    let video = document.querySelector("#videoElement");
    let canvas = document.querySelector("#canvasElement");
    let ctx = canvas.getContext('2d');
    photo = document.getElementById('photo');
    var localMediaStream = null;
    var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);
    
    
    
    function sendSnapshot() {
      if (!localMediaStream) {
        return;
      }
      ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 300, 150);
      let dataURL = canvas.toDataURL('image/jpeg');
      socket.emit('input-image', dataURL);
      console.log("IMAGE HAS BEEN EMITTED")

      socket.on('out-image-event',function(data){
        console.log("Image has been recieved")
        photo.setAttribute('src', data.image_data);
      }
      );
    }
    socket.on('connect', function() {
      console.log('Connected!');
    });
    var constraints = {
      video: {
        width: { min: 640 },
        height: { min: 480 }
      }
    };
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      video.srcObject = stream;
      localMediaStream = stream;
      setInterval(function () {
        sendSnapshot();
      }, 50);
    }).catch(function(error) {
      console.log(error);
    });
  });
