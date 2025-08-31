import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Accessing the database 
pc = pd.read_csv("/Users/harshsrivastava/Downloads/monthly_coffee_data.csv")
drc = pd.read_csv("/Users/harshsrivastava/Downloads/USD-INR.csv")
cmp = pd.read_csv("/Users/harshsrivastava/Downloads/Coffee Prices Historical Data/coffee-prices-historical-data.csv")

pc.columns = pc.columns.str.strip().str.lower()
drc.columns = drc.columns.str.strip().str.lower()
cmp.columns = cmp.columns.str.strip().str.lower()

# Coverting the dates to datetime objects and downsamples to a monthly average.
cmp['date'] = pd.to_datetime(cmp['date'], format='mixed', dayfirst=True, errors='coerce')
cmp_monthly = cmp.resample('M', on='date').mean().reset_index()
cmp_monthly.rename(columns={'value': 'CoffeePrice', 'date': 'Date'}, inplace=True)

pc['year-month'] = pd.to_datetime(pc['year-month'], format='mixed', dayfirst=True, errors='coerce')
pc.rename(columns={'year-month': 'Date'}, inplace=True)

drc['date'] = pd.to_datetime(drc['date'], format='mixed', dayfirst=True, errors='coerce')
drc_monthly = drc.resample('M', on='date').mean().reset_index()
drc_monthly.rename(columns={'date': 'Date', 'close': 'USDINR'}, inplace=True)

# Alliging all dates to a common monthly format.
cmp_monthly['Date'] = cmp_monthly['Date'].dt.to_period('M').dt.to_timestamp()
pc['Date'] = pc['Date'].dt.to_period('M').dt.to_timestamp()
drc_monthly['Date'] = drc_monthly['Date'].dt.to_period('M').dt.to_timestamp()

# It merges the three datasets into a single DataFrame.
df = cmp_monthly.merge(pc, on="Date", how="inner")
df = df.merge(drc_monthly[['Date', 'USDINR']], on="Date", how="inner")
df.dropna(inplace=True)

print("Merged DataFrame for prediction:")
print(df.head())
print("\nDataFrame shape:", df.shape)

# This defines the input features and the target variable.
features = ['production', 'consumption', 'USDINR']
target = 'CoffeePrice'

X = df[features]
y = df[target]

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluating the model's performance.
print("\n--- Model Evaluation ---")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# PLotting the actual versus predicted values to visualize the results.
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Prices', marker='o')
plt.plot(y_pred, label='Predicted Prices', marker='x', linestyle='--')
plt.title('Coffee Price Prediction - Simple Linear Regression')
plt.xlabel('Time Step')
plt.ylabel('Coffee Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# It predicts the price for the next month using the last known data point.
print("\n--- Next Month's Prediction ---")
last_known_values = df.iloc[-1][features].values.reshape(1, -1)
next_month_price = model.predict(last_known_values)[0]
print(f"Predicted Coffee Price for the next month: ${next_month_price:.2f}")

# By having the data for the next month we can hold up the decision-making process and wait for the right moment to purchase the coffee beans in bulk at a cheaper price