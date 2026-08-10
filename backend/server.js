const express = require("express");
const cors = require("cors");
const axios = require("axios");
const mongoose = require("mongoose");
require("dotenv").config();

const Prediction = require("./models/Prediction");

const app = express();

app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 5001;

// Connect to MongoDB
mongoose
    .connect(process.env.MONGO_URI)
    .then(() => {
        console.log("MongoDB connected");
    })
    .catch((error) => {
        console.error("MongoDB connection error:", error.message);
    });


// Home route
app.get("/", (req, res) => {
    res.json({
        message: "Fraud Detection Backend is running"
    });
});


// Prediction route
app.post("/api/predict", async (req, res) => {
    try {
        const transaction = req.body;

        // Send transaction to Flask ML API
        const response = await axios.post(
            "http://127.0.0.1:5000/predict",
            transaction
        );

        const result = response.data;

        // Save prediction to MongoDB
        const savedPrediction = await Prediction.create({
    prediction: result.prediction,
    fraudProbability: result.fraudProbability,
    amount: Number(transaction.Amount)
});

        res.json({
            prediction: result.prediction,
            fraudProbability: result.fraudProbability,
            predictionId: savedPrediction._id
        });

    } catch (error) {
        console.error("Prediction error:", error.message);

        res.status(500).json({
            message: "Prediction failed"
        });
    }
});

// Get prediction history
app.get("/api/predictions", async (req, res) => {
    try {
        const predictions = await Prediction
            .find()
            .sort({ createdAt: -1 });

        res.json(predictions);

    } catch (error) {
        console.error("History error:", error.message);

        res.status(500).json({
            message: "Failed to fetch prediction history"
        });
    }
});

app.listen(PORT, () => {
  console.log(`Backend running on port ${PORT}`);
});
