import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
columns = ['buying','maint','doors','person','lug_boot','safety','class']
df = pd.read_csv("car_evaluation/car.data", names=columns)

print("Sample data:\n", df.head())
print("\nClass distribution:\n", df['class'].value_counts())

#############################################################################

# 2. One-Hot Encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)  # avoids dummy variable trap
print("\nEncoded data sample:\n", df_encoded.head())

#############################################################################

# 3. Define features and target
X = df_encoded.drop('class_unacc', axis=1)  # pick one class column as reference
y = df_encoded['class_unacc']               # example: predict "unacc" vs others

# If you want multi-class classification instead of binary, keep original df['class']
# X = df_encoded.drop('class', axis=1)
# y = df['class']

#############################################################################

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#############################################################################

# 5. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#############################################################################

# 6. Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#############################################################################

# 7. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
