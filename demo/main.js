const express = require('express');
const fileUpload = require('express-fileupload');
const sharp = require('sharp');
const fs = require('fs');
const fireDetector = require('./fire-detector');

const app = express();

// Enable file upload middleware
app.use(fileUpload({
  useTempFiles: true,
  tempFileDir: '/tmp/',
}));

// Static files middleware
app.use(express.static('public'));

// Image upload and prediction route
app.post('/predict', async (req, res) => {
  if (!req.files || !req.files.image) {
    return res.status(400).send('No image was uploaded.');
  }

  let image = req.files.image;
  let imagePath = __dirname + '/public/' + image.name;
  let convertedImagePath = __dirname + '/public/converted_' + image.name;

  // Move the image to the server's public directory
  image.mv(imagePath, async function (err) {
    if (err) {
      return res.status(500).send(err);
    }

    try {
      let sharpInstance = sharp(imagePath);

      // Check image format
      if (image.mimetype === 'image/png') {
        // Convert PNG to three channels
        sharpInstance = sharpInstance.ensureAlpha().flatten();
      } else if (image.mimetype === 'image/jpeg' || image.mimetype === 'image/jpg') {
        // Convert JPEG to three channels
        sharpInstance = sharpInstance.ensureAlpha().toColorspace('srgb');
      } else {
        fs.unlinkSync(imagePath); // Remove the uploaded image
        return res.status(400).send('Unsupported image format.');
      }

      // Convert image to JPEG format
      sharpInstance = sharpInstance.toFormat('jpeg');

      // Save the converted image to a temporary file
      await sharpInstance.toFile(convertedImagePath);

      // Predict the fire probability
      let probability = await fireDetector.predict(convertedImagePath);

      // Remove the temporary converted image
      fs.unlinkSync(convertedImagePath);

      // Return the prediction result and image URL
      res.json({
        probability: probability,
        imageUrl: '/'+ image.name
      });

      // Remove the uploaded image
      fs.unlinkSync(imagePath);
    } catch (error) {
      console.error(error);
      fs.unlinkSync(imagePath); // Remove the uploaded image
      res.status(500).send('Error processing the image.');
    }
  });
});

app.listen(3000, () => console.log('Server started on port 3000'));
