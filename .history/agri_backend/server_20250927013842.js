// agri_backend/server.js

const express = require('express');
const dotenv = require('dotenv');
const connectDB = require('./config/db');
const authRoutes = require('./routes/authRoutes'); // <-- IMPORT the routes

dotenv.config();

connectDB();

const app = express();
app.use(express.json()); // <-- Add this middleware to parse JSON bodies

// Tell the app to use the authRoutes for any URL starting with /api/auth
app.use('/api/auth', authRoutes);

app.get('/', (req, res) => {
    res.send('Server is running and connected to the database.');
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});