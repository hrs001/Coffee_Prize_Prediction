# Coffee Price Prediction using Linear Regression

This project predicts **monthly coffee prices** based on historical coffee prices, production, consumption, and USD–INR exchange rates.  
The goal is to assist in **decision-making for bulk coffee purchases**, ensuring optimal timing to reduce costs and maximize profit.

## Project Workflow

1. **Data Sources**
   - `monthly_coffee_data.csv` → Coffee production & consumption.
   - `USD-INR.csv` → Historical USD-INR exchange rate.
   - `coffee-prices-historical-data.csv` → Global coffee price data.

2. **Data Cleaning & Preprocessing**
   - Standardized column names.  
   - Converted dates into monthly time-series format.  
   - Resampled to monthly averages.  
   - Merged datasets on common `Date` column.  

3. **Feature Engineering**
   - Features: `production`, `consumption`, `USDINR`.  
   - Target: `CoffeePrice`.  

4. **Modeling**
   - Used **Linear Regression**.  
   - Data split into **train (80%)** and **test (20%)**.  
   - Evaluation Metrics:
     - **RMSE (Root Mean Squared Error)**
     - **R² Score (Goodness of Fit)**  

5. **Visualization**
   - Plotted **actual vs predicted coffee prices**.  

6. **Prediction**
   - Predicted **next month’s coffee price** based on latest data.  

This prediction can help in inventory planning and purchasing strategies.
For example, a rising predicted price may encourage early bulk purchases, while a falling price may favor delayed procurement.
