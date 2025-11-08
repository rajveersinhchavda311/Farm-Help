const express = require('express');
const dotenv = require('dotenv');
const connectDB = require('./config/db');

// Load environment variables
dotenv.config();

// Connect to MongoDB - This is the only major task this file will perform
connectDB();

const app = express();

// A simple route to confirm the server is running
app.get('/', (req, res) => {
    res.send('Server is running and connected to the database.');
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});