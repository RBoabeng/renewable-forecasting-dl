# ⚡ Renewable Energy Microgrid Forecasting (Deep Learning)

## Project Overview

This project implements a Deep Learning pipeline to forecast renewable energy generation (Solar & Wind) for a localized microgrid. Accurate forecasting is critical for microgrid stability, allowing operators to optimize battery dispatching, reduce curtailment, and minimize reliance on external power grids.

The core model is a **Long Short-Term Memory (LSTM)** neural network built with PyTorch, designed to capture complex temporal dependencies in weather patterns and historical energy generation.

## Key Technical Features

* **Cyclical Time Encoding:** Engineered `hour_of_day` and `day_of_week` into continuous Sine/Cosine waves to prevent "Midnight Jump" discontinuities in the neural network.
* **Autoregressive Feedback:** Integrated historical target data ($t-1$ to $t-24$) alongside weather drivers to significantly improve peak-chasing behavior and shape tracking.
* **Dropout Regularization:** Mitigated model overfitting by tuning network capacity and implementing dropout layers.
* **Configuration-Driven Architecture:** All hyperparameters, data splits, and feature selections are controlled via a central `config.yaml`, ensuring code reproducibility without touching the core Python scripts.

##  Project Structure

```
renewable-forecasting-dl/
├── data/
│   ├── raw/                 # Raw datasets (weather, grid load, etc.)
│   └── processed/           # Cleaned data ready for modeling
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   └── 02_training.ipynb    # Model training, tuning, and evaluation
├── outputs/
│   ├── models/              # Saved PyTorch model weights (.pth)
│   └── scalers/             # Saved Scikit-Learn MinMax scalers (.pkl)
├── src/
│   ├── data_loader.py       # OOP-based data loading and sequencing
│   └── model.py             # PyTorch LSTM architecture
├── config.yaml              # Master control panel for features/hyperparameters
├── requirements.txt         # Python dependencies
└── README.md
```

## Getting Started

**1. Installation**

Clone the repository and install the required dependencies:

```
git clone [https://github.com/yourusername/renewable-forecasting-dl.git](https://github.com/yourusername/renewable-forecasting-dl.git)
cd renewable-forecasting-dl
pip install -r requirements.txt

```

**2. Configuration**

Adjust model hyperparameters or feature selections in `config.yaml`.


**3. Training & Evaluation**

Run the training pipeline via the interactive `notebooks/02_training.ipynb` to view real-time loss reduction, test splits, and plot Actual vs. Predicted energy outputs.


## Results & Learnings

* **The "Mean-Reversion" Challenge:** Early model iterations (predicting based purely on weather) defaulted to predicting the average energy output to minimize Mean Squared Error (MSE), failing to capture the extremes.

* **The Autoregressive Solution:** By introducing cyclical time encoding and the past 24 hours of actual generation, the model broke out of the mean-reversion trap, successfully tracking extreme peaks (>150 kW) and near-zero valleys.

* **Current Performance:** The model accurately tracks the generation shape curve, achieving an MAE of ~39 kW on unseen test data, exhibiting a slight autoregressive lag effect ($t-1$) common in step-forward forecasting.

## Future Work

* **Seq2Seq Architecture:** Upgrading the LSTM to an Encoder-Decoder model with Attention mechanisms to mitigate the $t-1$ lag effect.

* **Live Inference Script:** Building a predict.py endpoint that can consume live weather API data and dispatch commands to a simulated battery controller.

