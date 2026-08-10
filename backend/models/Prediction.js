const mongoose = require("mongoose");

const predictionSchema = new mongoose.Schema(
    {
        prediction: {
            type: String,
            required: true
        },

        fraudProbability: {
            type: Number,
            required: true
        },

        amount: {
            type: Number,
            required: true
        }
    },
    {
        timestamps: true
    }
);

const Prediction = mongoose.model(
    "Prediction",
    predictionSchema
);

module.exports = Prediction;