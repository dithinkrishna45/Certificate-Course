import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




# 1.load Dataset
columns = ['buying','maint','doors','person','lug_boot','safety','class']
df = pd.read_csv("car_evaluation/car.data",names=columns)

print(df.head())
print(df['class'].value_counts())

#############################################################################

# create a label encoder object
le = LabelEncoder()

# encode each column
for col in df.columns:
    df[col] = le.fit_transform(df[col])

print(df.head())

##########################################################################

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#######################################################################

model = LogisticRegression(max_iter=1000)  # increase iterations for convergence
model.fit(X_train, y_train)

#########################################################################

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#########################################################################

import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()