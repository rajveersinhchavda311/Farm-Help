// agri_backend/controllers/predictionController.js

const { PythonShell } = require('python-shell');
const path = require('path');
const fs = require('fs');

const makePrediction = async (req, res) => {
    const imagePath = req.file ? req.file.path : null;
    const inputParameters = req.body.inputParameters;

    if (!imagePath || !inputParameters) {
        if (imagePath) fs.unlinkSync(imagePath);
        return res.status(400).json({ message: 'Image file and inputParameters are required.' });
    }

    try {
        // --- Step 1: Define the paths to your Python scripts FIRST ---
        const detectionScriptPath = path.join(__dirname, '../../model/predict_disease.py');
        const riskScriptPath = path.join(__dirname, '../../model/predict_risk.py');
        
        console.log('Running detection model...');
        // --- Step 2: Run the first Python script ---
        const detectionArgs = [imagePath];
        const diseaseDetectionResult = await PythonShell.run(detectionScriptPath, {
            mode: 'json',
            args: detectionArgs
        });

        console.log('Running risk model...');
        // --- Step 3: Run the second Python script ---
        const riskArgs = [inputParameters];
        const diseaseRiskResult = await PythonShell.run(riskScriptPath, {
            mode: 'json',
            args: riskArgs
        });

        // --- Step 4: Clean up the uploaded image ---
        fs.unlinkSync(imagePath);
        console.log(`Cleaned up temporary file: ${imagePath}`);

        // --- Step 5: Send the combined results ---
        res.status(200).json({
            message: 'Prediction successful',
            diseaseDetection: diseaseDetectionResult[0],
            diseaseRisk: diseaseRiskResult[0]
        });

    } catch (error) {
        console.error("Error during Python script execution:", error);
        if (imagePath && fs.existsSync(imagePath)) {
            fs.unlinkSync(imagePath);
        }
        res.status(500).json({ message: 'Error during prediction process', error: error.message });
    }
};

module.exports = {
    makePrediction,
};