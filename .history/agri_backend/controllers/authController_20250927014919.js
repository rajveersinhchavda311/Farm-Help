// agri_backend/controllers/authController.js

// ... (keep the existing code for registerUser, loginUser, and generateToken)
const jwt = require('jsonwebtoken');
const User = require('../models/User');
require('dotenv').config();

const generateToken = (id) => {
    return jwt.sign({ id }, process.env.JWT_SECRET, { expiresIn: '1h' });
};

const registerUser = async (req, res) => {
    // ... (your existing registerUser function)
};

const loginUser = async (req, res) => {
    // ... (your existing loginUser function)
};

// --- Logic for getting the user's profile ---
const getUserProfile = async (req, res) => {
    if (req.user) {
        res.json({
            _id: req.user._id,
            username: req.user.username,
            email: req.user.email,
        });
    } else {
        res.status(404).json({ message: 'User not found' });
    }
};

// CRITICAL: Make sure `getUserProfile` is included in the exports
module.exports = {
    registerUser,
    loginUser,
    getUserProfile, // <-- Ensure this is here
};