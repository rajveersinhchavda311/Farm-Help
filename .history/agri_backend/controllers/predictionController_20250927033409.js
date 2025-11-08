const { PythonShell } = require('python-shell');
const path = require('path');
const fs = require('fs'); // The file system module to delete files

const makePrediction = async (req, res) => {
    const imagePath = req.file ? req.file.path : null;
    const inputParameters = req.body.inputParameters;

    // Validate that we received both the image and the text data
    if (!imagePath || !inputParameters) {
        if (imagePath) fs.unlinkSync(imagePath); // Clean up the orphaned image if it exists
        return res.status(400).json({ message: 'Image file and inputParameters are required.' });
    }

    try {
        // --- This is the options object for running Python scripts ---
        const options = {
            mode: 'json',
            // --- THIS IS THE CORRECTED LINE ---
            // The path now uses double backslashes (\\) to work correctly in a JavaScript string.
            pythonPath: 'C:\\Users\\mohit\\aptos_env\\Lib\\site-packages\\sympy\\printing\\python.exe',
            // ---------------------------------
        };

        // Define the paths to your two Python scripts
        const detectionScriptPath = path.join(__dirname, '../../model/predict_disease.py');
        const riskScriptPath = path.join(__dirname, '../../model/predict_risk.py');

        console.log(`Running image model on: ${imagePath}`);
        // Set the arguments for the first script (the image path)
        options.args = [imagePath];
        const diseaseDetectionResult = await PythonShell.run(detectionScriptPath, options);

        console.log(`Running risk model with data: ${inputParameters}`);
        // Change the arguments for the second script (the JSON string of parameters)
        options.args = [inputParameters];
        const diseaseRiskResult = await PythonShell.run(riskScriptPath, options);

        // After getting both predictions, delete the temporary uploaded image
        fs.unlinkSync(imagePath);

        // Send the combined results from both models back to the user
        res.status(200).json({
            message: 'Prediction successful',
            diseaseDetection: diseaseDetectionResult[0],
            diseaseRisk: diseaseRiskResult[0]
        });

    } catch (error) {
        console.error("Error during Python script execution:", error);
        // If an error occurs, still try to clean up the uploaded image
        if (imagePath && fs.existsSync(imagePath)) {
            fs.unlinkSync(imagePath);
        }
        res.status(500).json({ message: 'Error during prediction process', error: error.message || error });
    }
};

module.exports = {
    makePrediction,
};