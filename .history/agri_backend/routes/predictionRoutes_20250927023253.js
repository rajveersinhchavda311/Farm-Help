// agri_backend/routes/predictionRoutes.js

const express = require('express');
const { makePrediction } = require('../controllers/predictionController');
const { protect } = require('../middlewares/auth');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const router = express.Router();

const storage = multer.diskStorage({ /* ... your storage config ... */ });

// --- MODIFIED File filter to add a log ---
const fileFilter = (req, file, cb) => {
    // This log will tell us if the request passed the 'protect' middleware
    console.log(`Multer file filter is running for file: ${file.originalname}`);

    const allowedTypes = /jpeg|jpg|png/;
    const mimetype = allowedTypes.test(file.mimetype);
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());

    if (mimetype && extname) {
        return cb(null, true);
    }
    cb(new Error('Only images (jpeg, jpg, png) are allowed!'));
};

const upload = multer({
    storage: storage,
    fileFilter: fileFilter,
    limits: { fileSize: 5 * 1024 * 1024 }
});

router.post('/', protect, upload.single('image'), makePrediction);

module.exports = router;