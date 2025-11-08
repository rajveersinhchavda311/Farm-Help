const mongoose = require('mongoose');

const DiseaseReportSchema = new mongoose.Schema({
    diseaseName: {
        type: String,
        required: true
    },
    // GeoJSON Point for location: [longitude, latitude]
    location: {
        type: {
            type: String,
            enum: ['Point'],
            required: true
        },
        coordinates: {
            type: [Number], // Array of [longitude, latitude]
            required: true
        }
    },
    severity: { // e.g., confidence score from the model
        type: Number,
        required: true
    },
    reportedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    reportedAt: {
        type: Date,
        default: Date.now
    }
});

// CRITICAL: This enables fast geospatial queries for nearby farms
DiseaseReportSchema.index({ location: '2dsphere' });

module.exports = mongoose.model('DiseaseReport', DiseaseReportSchema);