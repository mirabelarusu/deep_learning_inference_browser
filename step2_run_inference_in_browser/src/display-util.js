function loadImage(src, onload) {
    var img = new Image();
    img.onload = onload;
    img.src = src;
    return img;
}

function showOverlayInCanvas(imgSrc, mskSrc, canvasID)
{
    const ctx = document.getElementById(canvasID).getContext('2d');  

    var img1 = loadImage(imgSrc, main);
    var img2 = loadImage(mskSrc, main);

    var imagesLoaded = 0;
    function main() {
        imagesLoaded += 1;

        if(imagesLoaded == 2) {
            // composite now
            ctx.drawImage(img1, 0, 0);

            ctx.globalAlpha = 1.0;
            ctx.drawImage(img2, 0, 0);
        }
    }
}


function showImgInCanvas(imgSrc, canvasID)
{
  const ctx = document.getElementById(canvasID).getContext('2d');
  var img = new Image();
  img.onload = function() {
      ctx.drawImage(img, 0, 0);
  };

  img.src=imgSrc;
}