// agri_backend/controllers/predictionController.js

// This function will be the main handler for our prediction route
const makePrediction = async (req, res) => {
    // req.file is an object containing file information, thanks to multer
    // req.body will contain the text fields, if there were any
    
    console.log('File received:', req.file);
    console.log('Body data received:', req.body);

    if (!req.file) {
        return res.status(400).json({ message: 'No image file uploaded.' });
    }
    
    // For now, we'll just send a success response.
    // Later, we'll add the Python script execution here.
    res.status(200).json({
        message: 'File and data received successfully. Ready for processing.',
        filename: req.file.filename,
        input_data: req.body.inputParameters
    });
};

module.exports = {
    makePrediction,
};