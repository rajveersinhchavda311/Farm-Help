// agri_backend/routes/predictionRoutes.js

const express = require('express');
const { makePrediction } = require('../controllers/predictionController');
const { protect } = require('../middlewares/auth');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const router = express.Router();

// --- Multer Configuration for storing uploaded files ---
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, '../public/uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, `${file.fieldname}-${Date.now()}${path.extname(file.originalname)}`);
    }
});

const upload = multer({ storage: storage });

// --- Define the Prediction Endpoint ---
// The request will first be checked by `protect` (for a valid token),
// then processed by `upload.single('image')` (to handle the file),
// and finally passed to our `makePrediction` controller.
router.post('/', protect, upload.single('image'), makePrediction);

module.exports = router;