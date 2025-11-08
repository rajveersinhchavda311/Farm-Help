// agri_backend/controllers/predictionController.js

const { PythonShell } = require('python-shell');
const path = require('path');
const fs = require('fs'); // We need the 'fs' module to delete files

const makePrediction = async (req, res) => {
    const imagePath = req.file ? req.file.path : null;
    const inputParameters = req.body.inputParameters;

    if (!imagePath || !inputParameters) {
        if (imagePath) fs.unlinkSync(imagePath); // Clean up uploaded image on error
        return res.status(400).json({ message: 'Image file and inputParameters are required.' });
    }

    try {
        // --- Step 1: Run Disease Detection Model (Image) ---
        // Note: You need a Python script that accepts arguments and prints JSON.
        // We will assume this script is named 'predict_disease.py' inside your 'model' folder.
        const detectionScriptPath = path.join(__dirname, '../../model/predict_disease.py');
        const detectionArgs = [imagePath];
        console.log('VERIFYING PATH for detection script:', detectionScriptPath);
        console.log('VERIFYING PATH for risk script:', riskScriptPath);
        const diseaseDetectionResult = await PythonShell.run(detectionScriptPath, {
            mode: 'json',
            args: detectionArgs
        });

        // --- Step 2: Run Disease Risk Model (Tabular Data) ---
        // We will assume this script is named 'predict_risk.py'
        const riskScriptPath = path.join(__dirname, '../../model/predict_risk.py');
        const riskArgs = [inputParameters]; // inputParameters is already a JSON string from form-data

        const diseaseRiskResult = await PythonShell.run(riskScriptPath, {
            mode: 'json',
            args: riskArgs
        });

        // --- Step 3: Clean up the uploaded image ---
        // After we get the predictions, we don't need the uploaded image anymore.
        fs.unlinkSync(imagePath);
        console.log(`Cleaned up temporary file: ${imagePath}`);

        // --- Step 4: Send the combined results to the user ---
        res.status(200).json({
            message: 'Prediction successful',
            diseaseDetection: diseaseDetectionResult[0], // python-shell returns an array of results
            diseaseRisk: diseaseRiskResult[0]
        });

    } catch (error) {
        console.error("Error during Python script execution:", error);
        if (imagePath && fs.existsSync(imagePath)) {
            fs.unlinkSync(imagePath); // Also clean up on error
        }
        res.status(500).json({ message: 'Error during prediction process', error: error.message });
    }
};

module.exports = {
    makePrediction,
};