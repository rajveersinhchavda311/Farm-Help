// agri_backend/routes/authRoutes.js

const express = require('express');
// Import getUserProfile from the controller
const { registerUser, loginUser, getUserProfile } = require('../controllers/authController');
// Import our new middleware
const { protect } = require('../middlewares/auth');

const router = express.Router();

router.post('/register', registerUser);
router.post('/login', loginUser);

// This route is protected. The `protect` function will run first.
// If the token is valid, it will then run `getUserProfile`.
router.get('/profile', protect, getUserProfile); // <-- ADD THIS ROUTE

module.exports = router;