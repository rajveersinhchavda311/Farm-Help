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
        const detectionScriptPath = path.join(__dirname, '../../model/predict_disease.py');
        const riskScriptPath = path.join(__dirname, '../../model/predict_risk.py');

        console.log(`Running image model on: ${imagePath}`);
        const diseaseDetectionResult = await PythonShell.run(detectionScriptPath, {
            mode: 'json',
            args: [imagePath]
        });

        console.log(`Running risk model with data: ${inputParameters}`);
        const diseaseRiskResult = await PythonShell.run(riskScriptPath, {
            mode: 'json',
            args: [inputParameters]
        });

        // Clean up the temporary uploaded image
        fs.unlinkSync(imagePath);

        // Send the combined results from both models back to the user
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
        res.status(500).json({ message: 'Error during prediction process', error: error.message || error });
    }
};

module.exports = {
    makePrediction,
};