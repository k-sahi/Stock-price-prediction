# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# ==============================================================================
#                      CORE LOGIC (from our notebook)
# ==============================================================================

# Use Streamlit's caching to avoid re-running functions unnecessarily
@st.cache_data
def create_features(df):
    """Creates technical analysis features on the dataframe."""
    df = df.copy()
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df


# The "Nuke and Rebuild" function for fetching data cleanly
@st.cache_data
def fetch_and_clean_data(ticker, period="10y"):
    """Fetches data and ensures it has a clean, simple column index."""
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    if data.empty:
        return None

    clean_df = pd.DataFrame(index=data.index)
    clean_df['Open'] = data['Open']
    clean_df['High'] = data['High']
    clean_df['Low'] = data['Low']
    clean_df['Close'] = data['Close']
    clean_df['Volume'] = data['Volume']
    return clean_df


# Function to train the model and return key metrics
def train_model(ticker_symbol):
    data = fetch_and_clean_data(ticker_symbol)
    if data is None:
        st.error(f"Could not fetch data for {ticker_symbol}. Please check the ticker.")
        return None, None, None, None

    featured_data = create_features(data)

    features_to_exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'target']
    features = [col for col in featured_data.columns if col not in features_to_exclude]
    X = featured_data[features]
    y = featured_data['target']

    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)

    # Evaluate on the last fold
    predictions = model.predict(X_test)
    class_report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    # Create Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Last Test Fold)')

    return model, features, fig, class_report


# Function to get the final prediction
def get_prediction(ticker, model, features):
    data = fetch_and_clean_data(ticker, period="100d")
    data_feat = data.copy()
    data_feat.ta.sma(length=20, append=True)
    data_feat.ta.ema(length=50, append=True)
    data_feat.ta.rsi(length=14, append=True)
    data_feat.ta.macd(fast=12, slow=26, signal=9, append=True)
    data_feat.ta.bbands(length=20, append=True)
    data_feat.ta.atr(length=14, append=True)
    data_feat.dropna(inplace=True)

    last_row = data_feat[features].iloc[[-1]]
    prediction = model.predict(last_row)[0]
    prediction_proba = model.predict_proba(last_row)[0]

    return prediction, prediction_proba, data['Close'].iloc[-1]


# ==============================================================================
#                           STREAMLIT UI
# ==============================================================================

st.set_page_config(layout="wide")
st.title("📈 Stock Price Prediction Agent")
st.write("Enter a stock ticker to train a model and get a prediction for the next trading day.")
st.info("⚠️ This is a learning project, not financial advice. Use at your own risk.")

# User Input
ticker_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, NVDA):", "TSLA").upper()

if st.button("Train and Predict"):
    if not ticker_symbol:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner(f"Training model for {ticker_symbol}... This may take a moment."):
            model, features, confusion_matrix_fig, class_report = train_model(ticker_symbol)

        if model:
            st.success("Model trained successfully!")

            # Display Results in two columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Performance")
                st.pyplot(confusion_matrix_fig)

            with col2:
                st.subheader("Classification Report")
                st.text(class_report)

            # --- Make and Display Final Prediction ---
            st.subheader("Live Prediction")
            with st.spinner("Fetching latest data for prediction..."):
                prediction, prediction_proba, current_price = get_prediction(ticker_symbol, model, features)

            if prediction == 1:
                st.metric(
                    label=f"Prediction for {ticker_symbol}",
                    value="UP 📈",
                    delta=f"{prediction_proba[1]:.2%} confidence"
                )
            else:
                st.metric(
                    label=f"Prediction for {ticker_symbol}",
                    value="DOWN 📉",
                    delta=f"{prediction_proba[0]:.2%} confidence",
                    delta_color="inverse"
                )
            st.write(f"Last close price: **${current_price:.2f}**")