import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset: house size (X) → house price (y)
x = np.array([500, 800, 1000, 1200, 1500]).reshape(-1,1)  # sqft
y = np.array([150000, 200000, 230000, 260000, 300000])    # price in $

model = LinearRegression()
model.fit(x,y)

# 2. Model parameters
print("Intercept (base price):", model.intercept_)
print("Slope (price per sqft):", model.coef_[0])   

size = 1100
predicted_price = model.predict([[size]])
print(f"Predicted price for {size} sqft = ${predicted_price[0]:.2f}")
