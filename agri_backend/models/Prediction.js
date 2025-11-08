const mongoose = require('mongoose');

// This defines the exact structure of your tabular input data
const InputParametersSchema = new mongoose.Schema({
    N: { type: Number, required: true }, // Nitrogen
    P: { type: Number, required: true }, // Phosphorus
    K: { type: Number, required: true }, // Potassium
    temperature: { type: Number, required: true },
    humidity: { type: Number, required: true },
    ph: { type: Number, required: true },
    rainfall: { type: Number, required: true }
}, { _id: false });

const PredictionSchema = new mongoose.Schema({
    user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    location: {
        type: { type: String, enum: ['Point'], required: true },
        coordinates: { type: [Number], required: true } // [longitude, latitude]
    },
    // This now perfectly matches your dataset columns
    inputParameters: InputParametersSchema,

    // The outputs from your two models
    diseaseDetectionOutput: { type: mongoose.Schema.Types.Mixed, required: true },
    diseaseRiskOutput: { type: mongoose.Schema.Types.Mixed, required: true }
}, {
    timestamps: true
});

PredictionSchema.index({ location: '2dsphere' });

module.exports = mongoose.model('Prediction', PredictionSchema);