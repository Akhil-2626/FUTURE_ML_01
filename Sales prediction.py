# ==========================================
# FUTURE SALES FORECASTING - SUPERSTORE DATA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    r"C:/Users/bhara/OneDrive/Desktop/akhil/Sample - Superstore.csv",
    encoding="latin1"
)

# Convert date column
df['Order Date'] = pd.to_datetime(df['Order Date'])

# ==========================================
# 2. AGGREGATE DAILY SALES
# ==========================================

daily_sales = ( 
    df.groupby('Order Date')['Sales']
    .sum()
    .reset_index()
)

daily_sales.columns = ['date', 'sales']

daily_sales = daily_sales.sort_values('date')

# ==========================================
# 3. CREATE TIME FEATURES
# ==========================================

daily_sales['year'] = daily_sales['date'].dt.year
daily_sales['month'] = daily_sales['date'].dt.month
daily_sales['day'] = daily_sales['date'].dt.day
daily_sales['dayofweek'] = daily_sales['date'].dt.dayofweek
daily_sales['weekofyear'] = daily_sales['date'].dt.isocalendar().week.astype(int)
daily_sales['quarter'] = daily_sales['date'].dt.quarter

# Seasonality
daily_sales['sin_month'] = np.sin(
    2*np.pi*daily_sales['month']/12
)

daily_sales['cos_month'] = np.cos(
    2*np.pi*daily_sales['month']/12
)

# Lag Features
daily_sales['lag_1'] = daily_sales['sales'].shift(1)
daily_sales['lag_7'] = daily_sales['sales'].shift(7)
daily_sales['lag_30'] = daily_sales['sales'].shift(30)

# Rolling Features
daily_sales['rolling_7'] = (
    daily_sales['sales']
    .rolling(7)
    .mean()
)

daily_sales['rolling_30'] = (
    daily_sales['sales']
    .rolling(30)
    .mean()
)

daily_sales = daily_sales.dropna()

# ==========================================
# 4. FEATURE SELECTION
# ==========================================

features = [
    'year',
    'month',
    'day',
    'dayofweek',
    'weekofyear',
    'quarter',
    'sin_month',
    'cos_month',
    'lag_1',
    'lag_7',
    'lag_30',
    'rolling_7',
    'rolling_30'
]

X = daily_sales[features]
y = daily_sales['sales']

# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

split_index = int(len(daily_sales)*0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ==========================================
# 6. TRAIN MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 7. PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# 8. EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-"*40)
print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R²  :", round(r2,4))

# ==========================================
# 9. FORECAST NEXT 90 DAYS
# ==========================================

future_days = 90

future_dates = pd.date_range(
    start=daily_sales['date'].max()+pd.Timedelta(days=1),
    periods=future_days,
    freq='D'
)

future_df = pd.DataFrame({
    'date': future_dates
})

future_df['year'] = future_df['date'].dt.year
future_df['month'] = future_df['date'].dt.month
future_df['day'] = future_df['date'].dt.day
future_df['dayofweek'] = future_df['date'].dt.dayofweek
future_df['weekofyear'] = future_df['date'].dt.isocalendar().week.astype(int)
future_df['quarter'] = future_df['date'].dt.quarter

future_df['sin_month'] = np.sin(
    2*np.pi*future_df['month']/12
)

future_df['cos_month'] = np.cos(
    2*np.pi*future_df['month']/12
)

last_sales = daily_sales['sales'].tail(30).mean()

future_df['lag_1'] = last_sales
future_df['lag_7'] = last_sales
future_df['lag_30'] = last_sales
future_df['rolling_7'] = last_sales
future_df['rolling_30'] = last_sales

future_sales = model.predict(
    future_df[features]
)

future_df['Predicted Sales'] = future_sales

print("\nFuture Forecast Sample:")
print(
    future_df[
        ['date','Predicted Sales']
    ].head()
)


# ==========================================
# CATEGORY-WISE FUTURE SALES FORECAST
# ==========================================

categories = df['Category'].unique()

category_forecasts = []

for category in categories:

    cat_df = df[df['Category'] == category]

    daily_cat = (
        cat_df.groupby('Order Date')['Sales']
        .sum()
        .reset_index()
    )

    daily_cat.columns = ['date', 'sales']

    if len(daily_cat) < 50:
        continue

    daily_cat['year'] = daily_cat['date'].dt.year
    daily_cat['month'] = daily_cat['date'].dt.month
    daily_cat['day'] = daily_cat['date'].dt.day
    daily_cat['dayofweek'] = daily_cat['date'].dt.dayofweek

    X_cat = daily_cat[['year','month','day','dayofweek']]
    y_cat = daily_cat['sales']

    cat_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    cat_model.fit(X_cat, y_cat)

    future_dates = pd.date_range(
        start=daily_cat['date'].max() + pd.Timedelta(days=1),
        periods=30,
        freq='D'
    )

    future_cat = pd.DataFrame({
        'date': future_dates
    })

    future_cat['year'] = future_cat['date'].dt.year
    future_cat['month'] = future_cat['date'].dt.month
    future_cat['day'] = future_cat['date'].dt.day
    future_cat['dayofweek'] = future_cat['date'].dt.dayofweek

    preds = cat_model.predict(
        future_cat[['year','month','day','dayofweek']]
    )

    category_forecasts.append({
        'Category': category,
        'Forecasted Sales': preds.sum()
    })

forecast_category_df = pd.DataFrame(category_forecasts)

print(forecast_category_df)


# ==========================================
# 10. BUSINESS FRIENDLY VISUALIZATION
# ==========================================

fig, ax = plt.subplots(
    2,
    2,
    figsize=(18,10)
)

# Historical Sales
ax[0,0].plot(
    daily_sales['date'],
    daily_sales['sales']
)
ax[0,0].set_title(
    "Historical Sales Trend"
)

# Actual vs Predicted
ax[0,1].plot(
    y_test.values,
    label="Actual"
)

ax[0,1].plot(
    predictions,
    label="Predicted"
)

ax[0,1].legend()

ax[0,1].set_title(
    "Actual vs Predicted Sales"
)

# Future Forecast
ax[1,0].plot(
    future_df['date'],
    future_df['Predicted Sales']
)

ax[1,0].set_title(
    "Next 90 Days Sales Forecast"
)

# Monthly Sales Pattern
monthly_sales = (
    df.groupby(
        df['Order Date'].dt.month
    )['Sales']
    .sum()
)

ax[1,1].bar(
    monthly_sales.index,
    monthly_sales.values
)

ax[1,1].set_title(
    "Monthly Sales Seasonality"
)

plt.tight_layout()
plt.show()

# ==========================================
# 11. FORECAST TABLE
# ==========================================

forecast_df = future_df[
    ['date', 'Predicted Sales']
]

print("\nForecast Table")
print(forecast_df.head(20))


# ==========================================
# FUTURE CATEGORY SALES VISUALIZATION
# ==========================================

forecast_category_df = forecast_category_df.sort_values(
    by='Forecasted Sales',
    ascending=False
)

plt.figure(figsize=(10,6))

bars = plt.bar(
    forecast_category_df['Category'],
    forecast_category_df['Forecasted Sales']
)

plt.title(
    'Predicted Category Sales for Next 30 Days',
    fontsize=14
)

plt.xlabel('Category')
plt.ylabel('Forecasted Sales')

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f'{bar.get_height():.0f}',
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show() 