// agri_backend/routes/authRoutes.js

const express = require('express');
const { registerUser, loginUser } = require('../controllers/authController');

const router = express.Router();

// When a POST request comes to /api/auth/register, run the registerUser function
router.post('/register', registerUser);

// When a POST request comes to /api/auth/login, run the loginUser function
router.post('/login', loginUser);

// This line makes the router available to the main server file
module.exports = router;