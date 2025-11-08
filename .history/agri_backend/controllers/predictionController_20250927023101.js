// agri_backend/controllers/predictionController.js

const makePrediction = async (req, res) => {
    // Thanks to multer, the uploaded file's info is in `req.file`
    // The text data from the form-data is in `req.body`
    
    console.log('File successfully received:', req.file);
    console.log('Body data successfully received:', req.body);

    if (!req.file) {
        return res.status(400).json({ message: 'No image file uploaded.' });
    }
    
    // We are not running the Python model yet. 
    // This is just a success response to confirm the upload worked.
    res.status(200).json({
        message: 'File and data received successfully. Ready for processing.',
        filename: req.file.filename,
    });
};

module.exports = {
    makePrediction,
};