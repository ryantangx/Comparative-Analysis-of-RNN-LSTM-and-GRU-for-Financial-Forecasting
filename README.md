# A Comparative Analysis of RNN Architectures for Multivariate Financial Forecasting

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Abstract
[cite_start]Financial market forecasting is characterized by a high degree of noise, making precise price prediction an intractable problem[cite: 93]. [cite_start]This project reframes the task as a binary classification of price direction ("Up" or "Down") and provides a rigorous comparative analysis of three recurrent neural network architectures: **Simple RNN, LSTM, and GRU**[cite: 94].

The models were trained on a multivariate, 10-year (2014-2023) dataset of four technology stocks (AAPL, META, NFLX, TSLA). [cite_start]The results demonstrate that while the baseline Simple RNN failed to generalize due to the vanishing gradient problem, the gated architectures (LSTM and GRU) captured predictive signals, achieving an AUC of ~0.56[cite: 95, 96, 97].

---

## 📂 Data & Preprocessing
[cite_start]The dataset comprises 10 years of daily OHLC data from **2014-01-02 to 2023-12-29**[cite: 140].
* [cite_start]**Target Variable:** Binary classification where `1` = Bullish (Next Day Close > Current Close) and `0` = Bearish/Neutral[cite: 146].
* [cite_start]**Feature Set (18 Total):** Adjusted Close, Volume, RSI (14-day), MACD, SMA (50/100), and EMA[cite: 141].

### Preprocessing Pipeline
1.  [cite_start]**Chronological Split:** Train (2014-2021), Validation (2022), Test (2023) to prevent lookahead bias[cite: 168].
2.  [cite_start]**Scaling:** MinMaxScaler (0, 1) fit on training data only[cite: 170].
3.  [cite_start]**Windowing:** A sliding window of **60 timesteps** (60 trading days) was used to capture sequential patterns[cite: 175].

---

## 📈 Exploratory Data Analysis (EDA)
[cite_start]An extensive EDA was conducted to identify the primary characteristics, temporal dependencies, and inter-relationships within the multivariate dataset[cite: 153].

### 1. Adjusted Close Prices (Trend Analysis)
[cite_start]The visualization of adjusted close prices on a logarithmic scale highlights the diverse growth trajectories and significant volatility differentials between the four assets (AAPL, META, NFLX, TSLA) over the decade[cite: 154].

![Adjusted Close Prices](adj_close_log_scale.png)
[cite_start]*(Figure 1: Adjusted Close Prices (Log Scale) [cite: 289])*

### 2. Daily Returns (Stationarity Check)
[cite_start]Unlike the raw price series which exhibits non-stationary trends, the daily percent returns are demonstrably stationary, oscillating around a mean of zero[cite: 155]. [cite_start]This property is fundamental for time-series modeling, validating the choice to predict price direction rather than raw price levels[cite: 156, 157].

![Daily Returns](daily_returns.png)
[cite_start]*(Figure 2: Daily Percent Returns [cite: 291])*

### 3. Correlation Matrix (Multivariate Justification)
[cite_start]The Pearson correlation matrix reveals moderate to strong positive correlations (0.51 to 0.72) across all asset pairs[cite: 158]. [cite_start]This indicates that the assets tend to move in sync with broader market sentiment, justifying a multivariate modeling approach to capture these shared patterns[cite: 159, 160].

![Correlation Matrix](returns_correlation_heatmap.png)
[cite_start]*(Figure 3: Correlation Matrix of Daily Returns [cite: 293])*

---

## 🧠 Model Architectures
[cite_start]Three distinct models were developed alongside a Dummy Classifier baseline[cite: 184]. All neural networks shared a consistent architecture to ensure a fair comparison:

* **Input:** (60 time steps, 18 features).
* [cite_start]**Recurrent Layer:** 50 units (SimpleRNN, LSTM, or GRU) with Tanh activation[cite: 186].
* **Regularization:** Dropout (0.2).
* **Dense Layer:** 25 units (ReLU).
* **Output:** 1 unit (Sigmoid).
* [cite_start]**Optimization:** Adam optimizer with Binary Crossentropy loss[cite: 187].

---

## 📊 Results & Performance
The models were evaluated on the unseen Test Set (2023 data). [cite_start]The **Area Under the ROC Curve (AUC)** was the primary metric used to assess discriminative ability[cite: 192].

| Model | AUC | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM** | **0.5599** | 48.55% | 0.4855 | 1.0000 | 0.6537 | 41.38 |
| **GRU** | **0.5568** | 48.55% | 0.4855 | 1.0000 | 0.6537 | 55.90 |
| **Simple RNN** | 0.5107 | 49.87% | 0.4464 | 0.1355 | 0.2079 | 20.53 |
| **Dummy** | 0.5000 | 51.45% | 0.0000 | 0.0000 | 0.0000 | 0.00 |
*[Data Source: Table 1, cite: 195]*

### 1. Receiver Operating Characteristic (ROC) Curve
The ROC curve visually confirms that LSTM and GRU (top lines) possess discriminative power significantly better than the Simple RNN, which hovers near the "No Skill" diagonal.

![ROC Curve](roc_curves.png)
[cite_start]*(Figure 5: Receiver Operating Characteristic Curve [cite: 297])*

### 2. Training History (Vanishing Gradient)
[cite_start]The training history reveals a critical insight: The **Simple RNN's validation loss flatlined immediately**[cite: 202]. This is a classic demonstration of the **vanishing gradient problem**, where the model failed to learn long-range dependencies from the 60-day sequences. [cite_start]In contrast, LSTM and GRU showed a clear learning curve[cite: 203, 204].

![Training History](training_history_loss.png)
[cite_start]*(Figure 6: Model Training & Validation Loss [cite: 299])*

---

## 🔍 Key Insights & Discussion

1.  **Superiority of Gated Architectures:** The LSTM (AUC 0.5599) and GRU (AUC 0.5568) significantly outperformed the Simple RNN. [cite_start]Their gating mechanisms successfully allowed error signals to backpropagate through time, capturing patterns the Simple RNN missed[cite: 196, 206].
2.  **High Recall / Low Precision Bias:** The LSTM and GRU models achieved perfect Recall (1.0) but low Precision (~0.48). [cite_start]This indicates the models learned to identify "Bullish" trends but were overly optimistic, predicting "Up" too frequently[cite: 211].
3.  **Financial Feasibility:** While an AUC of 0.56 indicates predictive power better than random chance, the models are not yet suitable for standalone algorithmic trading. [cite_start]They serve better as confirmatory indicators within a larger decision-making system[cite: 231, 233].

## 🚀 Future Improvements
* [cite_start]**Threshold Optimization:** Moving away from the default 0.5 classification threshold to maximize precision[cite: 237].
* [cite_start]**Exogenous Data:** Incorporating market sentiment, news, and macroeconomic indicators to improve signal-to-noise ratio[cite: 240].

---
*© 2025 Ryan Tang.*
