import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ------------------- DATA -------------------

# 1. Load dataset
df = pd.read_csv("DataSet/50_startups_sample.csv", encoding='latin1')

# Encode categorical 'State'
df = pd.get_dummies(df, columns=["State"], drop_first=True)
print("Training columns:", df.columns)

# 2. Features (X) and Target (y)
X = df.drop(columns=["Profit"])
y = df["Profit"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------- MODEL -------------------

# 4. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# ------------------- EVALUATION -------------------

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.show()

# ------------------- USER PREDICTION -------------------

print("\n--- Predict Profit ---")
rd = float(input("Enter R&D Spend: "))
adm = float(input("Enter Administration Spend: "))
mkt = float(input("Enter Marketing Spend: "))
state = input("Enter State (New York / California / Florida): ")

#  Match training dummy columns exactly
user_data = {
    "R&D Spend": rd,
    "Administration": adm,
    "Marketing Spend": mkt,
    "State_New York": 1 if state == "New York" else 0,
    "State_Florida": 1 if state == "Florida" else 0
}
user_df = pd.DataFrame([user_data])

# Make sure order of columns matches training set
user_df = user_df.reindex(columns=X.columns, fill_value=0)

# Predict profit
predicted_profit = model.predict(user_df)[0]
print(f"\nPredicted Profit: {predicted_profit:.2f}")