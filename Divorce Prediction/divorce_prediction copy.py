import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("Loading data...")
data = pd.read_csv('Divorce Prediction/marriage_data.csv')

# Basic data exploration
print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nTarget variable distribution:")
print(data['divorced'].value_counts())
print(f"Divorce rate: {data['divorced'].mean():.2%}")

# Check for missing values
print(f"\nMissing values: {data.isnull().sum().sum()}")

# Prepare the data
print("\nPreparing data...")

# Create a copy for processing
df = data.copy()

# Encode categorical variables
categorical_columns = ['education_level', 'employment_status', 'religious_compatibility', 
                      'conflict_resolution_style', 'marriage_type']

for column in categorical_columns:
    if column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        print(f"Encoded {column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Separate features and target
X = df.drop('divorced', axis=1)
y = df['divorced']

print(f"\nFeatures: {list(X.columns)}")
print(f"Number of features: {X.shape[1]}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n{'='*50}")
print("MODEL RESULTS")
print(f"{'='*50}")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Not Divorced', 'Divorced']))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Feature importance
feature_names = X.columns
importances = rf_model.feature_importances_

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:<25} : {row['importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 8))
top_15_features = feature_importance_df.head(15)
sns.barplot(data=top_15_features, x='importance', y='feature', palette='viridis')
plt.title('Top 15 Feature Importances - Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Not Divorced', 'Divorced'],
           yticklabels=['Not Divorced', 'Divorced'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Example prediction for new data
print(f"\n{'='*50}")
print("EXAMPLE PREDICTION")
print(f"{'='*50}")

# Create a sample person (you can modify these values)
sample_data = {
    'age_at_marriage': 28,
    'marriage_duration_years': 5,
    'num_children': 2,
    'education_level': 1,  # Encoded: Bachelor = 1 (check encoding above)
    'employment_status': 1,  # Encoded: Full-time = 1
    'combined_income': 65000,
    'religious_compatibility': 2,  # Encoded: Same Religion = 2
    'cultural_background_match': 1,
    'communication_score': 7.0,
    'conflict_frequency': 2,
    'conflict_resolution_style': 1,  # Encoded: Collaborative = 1
    'financial_stress_level': 4.0,
    'mental_health_issues': 0,
    'infidelity_occurred': 0,
    'counseling_attended': 0,
    'social_support': 7.0,
    'shared_hobbies_count': 3,
    'marriage_type': 1,  # Encoded: Love = 1
    'pre_marital_cohabitation': 1,
    'domestic_violence_history': 0,
    'trust_score': 8.0
}

# Convert to dataframe
sample_df = pd.DataFrame([sample_data])

# Make prediction
sample_prediction = rf_model.predict(sample_df)[0]
sample_probabilities = rf_model.predict_proba(sample_df)[0]

print(f"Sample person prediction: {'Divorced' if sample_prediction == 1 else 'Not Divorced'}")
print(f"Probability of divorce: {sample_probabilities[1]:.2%}")
print(f"Probability of staying married: {sample_probabilities[0]:.2%}")

# Model insights
print(f"\n{'='*50}")
print("KEY INSIGHTS")
print(f"{'='*50}")
print(f"• Most important feature: {feature_importance_df.iloc[0]['feature']}")
print(f"• Top 3 features account for {feature_importance_df.head(3)['importance'].sum():.1%} of importance")
print(f"• Model accuracy on test data: {test_accuracy:.1%}")
print(f"• Total number of decision trees: {rf_model.n_estimators}")

print("\nAnalysis completed!")