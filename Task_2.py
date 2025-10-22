# 🧠 Stock Price Prediction Project (Tesla) - Data Science Internship
# Company: Arch Technologies
# Intern: [Abdullah Umar]

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime


plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#0e1117",
    "figure.facecolor": "#0e1117",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333333"
})
pio.templates.default = "plotly_dark"



# Load Dataset
df = pd.read_csv("C:/Users/Abdullah Umer/Desktop/Arch Technologies Internship/Task 2/TESLA.csv")
print("✅ Dataset loaded successfully!\n")
print(df.info())
print("\nFirst few rows:\n", df.head())


# Data Preprocessing 
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())


# Feature Engineering 
df['Return'] = df['Close'].pct_change()
df['MA7'] = df['Close'].rolling(window=7).mean()
df['MA21'] = df['Close'].rolling(window=21).mean()
df['20SD'] = df['Close'].rolling(window=20).std()
df['UpperBand'] = df['MA21'] + (df['20SD'] * 2)
df['LowerBand'] = df['MA21'] - (df['20SD'] * 2)
df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
df['Momentum'] = df['Close'] - 1
df['Volatility'] = df['Return'].rolling(window=20).std()
df.dropna(inplace=True)




# Visualization Section 
print("\n📊 Creating visualizations...")

# 1. Line plot - Closing Price
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], color='cyan', label='Close Price')
plt.title('Tesla Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# 2. MA7, MA21 and Bollinger Bands
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close', color='white')
plt.plot(df['MA7'], label='MA7', color='red')
plt.plot(df['MA21'], label='MA21', color='green')
plt.fill_between(df.index, df['LowerBand'], df['UpperBand'], color='gray', alpha=0.3)
plt.title('Moving Averages & Bollinger Bands')
plt.legend()
plt.show()

# 3. Volume
plt.figure(figsize=(12,5))
plt.bar(df.index, df['Volume'], color='orange')
plt.title('Tesla Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()

# 4. Histogram of Returns
plt.figure(figsize=(8,5))
plt.hist(df['Return'], bins=50, color='purple', alpha=0.8)
plt.title('Histogram of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

# 5. Boxplot of Returns by Month
df['Month'] = df.index.month
plt.figure(figsize=(10,6))
sns.boxplot(x='Month', y='Return', data=df, palette='mako', legend=False)
plt.title('Monthly Return Distribution')
plt.show()

# 6. Scatter plot: Volume vs Return
plt.figure(figsize=(8,6))
plt.scatter(df['Volume'], df['Return'], color='yellow', alpha=0.6)
plt.xscale('log')
plt.title('Volume vs Return (log scale)')
plt.xlabel('Volume')
plt.ylabel('Return')
plt.show()

# 7. Rolling Volatility
plt.figure(figsize=(12,6))
plt.plot(df['Volatility'], color='lime', label='20-day Volatility')
plt.title('Rolling Volatility')
plt.legend()
plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# 9. Cumulative Returns
df['Cumulative Return'] = (1 + df['Return']).cumprod()
plt.figure(figsize=(12,6))
plt.plot(df['Cumulative Return'], color='cyan')
plt.title('Cumulative Return Over Time')
plt.show()

# 10. Lag plot
pd.plotting.lag_plot(df['Close'], lag=1)
plt.title('Lag Plot of Close Price')
plt.show()

# 11. Density Plot
plt.figure(figsize=(10,5))
sns.kdeplot(df['Return'], fill=True, color='magenta')
plt.title('Density Plot of Returns')
plt.show()

# 12. Scatter Close vs MA21
plt.figure(figsize=(8,6))
plt.scatter(df['MA21'], df['Close'], color='skyblue', alpha=0.6)
plt.title('Close vs 21-Day MA')
plt.xlabel('MA21')
plt.ylabel('Close')
plt.show()

# 13. Interactive Plotly Visualization
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.03, subplot_titles=('Tesla Stock Price', 'Volume'),
                    row_heights=[0.7, 0.3])
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='cyan')), row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='purple'), row=2, col=1)
fig.update_layout(template='plotly_dark', height=700, title_text='Tesla Stock Price & Volume')
fig.show()



# Modeling
print("\n⚙️ Preparing data for models...")

# Shift target for prediction
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21', 'EMA', 'Volatility']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression 
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Random Forest 
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# LSTM Model 
scaler_lstm = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler_lstm.fit_transform(df[['Close']])

X_lstm, y_lstm = [], []
for i in range(60, len(scaled_data)):
    X_lstm.append(scaled_data[i-60:i, 0])
    y_lstm.append(scaled_data[i, 0])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# Split train/test for LSTM
train_size = int(len(X_lstm)*0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, verbose=1)

predictions_lstm = model.predict(X_test_lstm)
predictions_lstm = scaler_lstm.inverse_transform(predictions_lstm)

# Evaluation 
def evaluate(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n📈 {model_name} Results:")
    print(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# Plot Predictions 
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual', color='cyan')
plt.plot(y_pred_rf, label='Predicted (RF)', color='yellow')
plt.title('Actual vs Predicted Prices (Random Forest)')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual', color='cyan')
plt.plot(y_pred_lr, label='Predicted (LR)', color='red')
plt.title('Actual vs Predicted Prices (Linear Regression)')
plt.legend()
plt.show()

print("\n✅ Stock Price Prediction Project Completed Successfully!")








