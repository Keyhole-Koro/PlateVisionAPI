const express = require('express');
const multer = require('multer');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const app = express();
const port = 3000;

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve the index.html file at the root URL
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Set up multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Handle image upload
app.post('/upload', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).send({ message: 'No file uploaded' });
  }

  try {
    const form = new FormData();
    form.append('file', req.file.buffer, req.file.originalname);
    form.append('measure', 'true');

    // Use the service name "recognition-api" to connect to the recognition API
    const response = await axios.post('http://recognition-api:8000/process_image/', form, {
        headers: {
        ...form.getHeaders(),
        'accept': 'application/json',
      },
    });

    res.send({ message: 'Image uploaded and processed successfully', result: response.data });
  } catch (error) {
    console.error(`Error processing image: ${error.message}`);
    res.status(500).send({ message: 'Failed to process image' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});