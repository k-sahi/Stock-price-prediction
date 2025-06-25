# 📈 Stock Price Prediction Agent

This is a web application built with Streamlit that uses a Random Forest machine learning model to predict the direction of the next day's stock price.

**Disclaimer:** This project is for educational purposes only and should not be used as financial advice. The stock market is highly unpredictable, and the model's predictions are not a guarantee of future performance.

---

### Features

-   Enter any valid stock ticker from Yahoo Finance.
-   Trains a new `RandomForestClassifier` on 10 years of historical data for that ticker.
-   Engineers technical analysis features (SMA, EMA, RSI, MACD, etc.) to feed the model.
-   Displays model performance metrics, including a Confusion Matrix and a Classification Report.
-   Provides a final prediction (UP or DOWN) for the next trading day, along with the model's confidence level.

### Technology Stack

-   **Python 3.9+**
-   **Streamlit:** For the web application interface.
-   **scikit-learn:** For the Random Forest model and evaluation metrics.
-   **yfinance:** To download historical stock data from Yahoo Finance.
-   **pandas-ta:** For calculating technical analysis indicators.
-   **Pandas:** For data manipulation.
-   **Seaborn & Matplotlib:** For plotting the confusion matrix.

---

### Setup and Installation

Follow these steps to run the application on your local machine.

**1. Clone the repository:**
```bash
git clone <your-repository-url>
cd stock-prediction-app
```

**2. Create and activate a virtual environment:**

*   **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
*   **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

**3. Install the required libraries:**
```bash
pip install -r requirements.txt
```

---

### Usage

Once the setup is complete, run the following command in your terminal:

```bash
streamlit run app.py
```

Your web browser should automatically open with the application running.

---

### Screenshot

*(Here you can add a screenshot of your running application)*

![App Screenshot](path/to/your/screenshot.png)