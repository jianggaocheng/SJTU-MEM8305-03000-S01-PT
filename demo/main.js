const express = require('express');
const fileUpload = require('express-fileupload');
const fireDetector = require('./fire-detector');

const app = express();

// Enable file upload middleware
app.use(fileUpload({
  useTempFiles : true,
  tempFileDir : '/tmp/',
}));

// Static files middleware
app.use(express.static('public'))

// Image upload and prediction route
app.post('/predict', async (req, res) => {
  if (!req.files || !req.files.image) {
    return res.status(400).send('No image was uploaded.');
  }

  let image = req.files.image;
  let imagePath = './public/' + image.name;
  
  // Move the image to the server
  image.mv(imagePath, async function(err) {
    if (err) {
      return res.status(500).send(err);
    }

    // Predict the fire probability
    let probability = await fireDetector.predict(imagePath);

    // Return the prediction result and image URL
    res.json({
      probability: probability,
      imageUrl: '/' + image.name
    });
  });
});

app.listen(3000, () => console.log('Server started on port 3000'));
