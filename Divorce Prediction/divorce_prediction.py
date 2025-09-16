import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

class DivorcePredictor:
    def __init__(self, data_path=None, data=None):
        """
        Initialize the DivorcePredictor class
        
        Args:
            data_path (str): Path to CSV file
            data (pd.DataFrame): Pandas DataFrame with the data
        """
        if data is not None:
            self.data = data
        elif data_path:
            self.data = pd.read_csv("Divorce Prediction/marriage_data.csv", encoding='latin1')
        else:
            raise ValueError("Either data_path or data must be provided")
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("Dataset Shape:", self.data.shape)
        print("\nDataset Info:")
        print(self.data.info())
        print("\nFirst 5 rows:")
        print(self.data.head())
        print("\nTarget variable distribution:")
        print(self.data['divorced'].value_counts())
        print("\nMissing values:")
        print(self.data.isnull().sum().sum())
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        # Divorce rate
        divorce_rate = self.data['divorced'].mean()
        print(f"\nOverall Divorce Rate: {divorce_rate:.2%}")
        
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("Preprocessing data...")
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Identify categorical columns that need encoding
        categorical_columns = ['education_level', 'employment_status', 'religious_compatibility', 
                             'conflict_resolution_style', 'marriage_type']
        
        # Encode categorical variables
        for column in categorical_columns:
            if column in df.columns:
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                self.label_encoders[column] = le
        
        # Separate features and target
        X = df.drop('divorced', axis=1)
        y = df['divorced']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features (optional for Random Forest, but can help with interpretability)
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        self.X_train_scaled[numerical_columns] = self.scaler.fit_transform(self.X_train[numerical_columns])
        self.X_test_scaled[numerical_columns] = self.scaler.transform(self.X_test[numerical_columns])
        
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Test set size: {self.X_test.shape[0]} samples")
        print(f"Number of features: {self.X_train.shape[1]}")
        
    def train_model(self, use_grid_search=True):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        if use_grid_search:
            # Grid search for hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Use a smaller parameter grid for faster execution
            param_grid_small = {
                'n_estimators': [100, 200],
                'max_depth': [None, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid_small, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            self.model = grid_search.best_estimator_
            print("Best parameters:", grid_search.best_params_)
            print("Best cross-validation score:", grid_search.best_score_)
            
        else:
            # Use default parameters with some optimization
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(self.X_train, self.y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Training and test accuracy
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # ROC AUC Score
        roc_auc = roc_auc_score(self.y_test, y_test_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(self.y_test, y_test_pred, 
                                  target_names=['Not Divorced', 'Divorced']))
        
        # Confusion Matrix
        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(cm)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': classification_report(self.y_test, y_test_pred, output_dict=True)
        }
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance"""
        feature_names = self.X_train.columns
        importances = self.model.feature_importances_
        
        # Create DataFrame for easier handling
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(top_n)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
        print(f"\nTop {top_n} Most Important Features:")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<25} : {row['importance']:.4f}")
        
        return feature_importance_df
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix heatmap"""
        y_test_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Divorced', 'Divorced'],
                   yticklabels=['Not Divorced', 'Divorced'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_test_proba)
        roc_auc = roc_auc_score(self.y_test, y_test_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def predict_new_sample(self, sample_data):
        """
        Predict divorce probability for new data
        
        Args:
            sample_data (dict): Dictionary with feature values
            
        Returns:
            tuple: (prediction, probability)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create DataFrame from sample data
        sample_df = pd.DataFrame([sample_data])
        
        # Encode categorical variables
        for column, encoder in self.label_encoders.items():
            if column in sample_df.columns:
                sample_df[column] = encoder.transform(sample_df[column].astype(str))
        
        # Make prediction
        prediction = self.model.predict(sample_df)[0]
        probability = self.model.predict_proba(sample_df)[0]
        
        return prediction, probability
    
    def get_model_insights(self):
        """Get insights about the model and data"""
        print("\n" + "="*50)
        print("MODEL INSIGHTS")
        print("="*50)
        
        # Feature importance insights
        feature_importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Key findings:")
        print(f"• Most important feature: {feature_importance_df.iloc[0]['feature']}")
        print(f"• Top 3 features account for {feature_importance_df.head(3)['importance'].sum():.2%} of total importance")
        
        # Model parameters
        print(f"• Model uses {self.model.n_estimators} decision trees")
        print(f"• Maximum tree depth: {self.model.max_depth}")
        print(f"• Out-of-bag score: {getattr(self.model, 'oob_score_', 'N/A')}")


def main():
    """Main function to run the complete analysis"""
    print("DIVORCE PREDICTION USING RANDOM FOREST")
    print("="*50)
    
    # For this example, we'll create sample data since we have the CSV content
    # In practice, you would load from a CSV file:
    # predictor = DivorcePredictor(data_path='your_data.csv')
    
    # Create sample data (you should replace this with your actual data loading)
    # Since we have the data content, let's assume it's already loaded
    try:
        # Try to load from CSV file if it exists
        predictor = DivorcePredictor(data_path='marriage_data.csv')
    except:
        # If file doesn't exist, you'll need to create it from your data
        print("Please save your data as 'marriage_data.csv' and run again.")
        print("Or modify the code to work with your specific data source.")
        return
    
    # Perform analysis
    predictor.explore_data()
    predictor.preprocess_data()
    predictor.train_model(use_grid_search=True)
    
    # Evaluate model
    results = predictor.evaluate_model()
    
    # Visualizations
    predictor.plot_feature_importance()
    predictor.plot_confusion_matrix()
    predictor.plot_roc_curve()
    
    # Get insights
    predictor.get_model_insights()
    
    # Example prediction for new data
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    sample_person = {
        'age_at_marriage': 28,
        'marriage_duration_years': 5,
        'num_children': 2,
        'education_level': 'Bachelor',
        'employment_status': 'Full-time',
        'combined_income': 65000,
        'religious_compatibility': 'Same Religion',
        'cultural_background_match': 1,
        'communication_score': 7.0,
        'conflict_frequency': 2,
        'conflict_resolution_style': 'Collaborative',
        'financial_stress_level': 4.0,
        'mental_health_issues': 0,
        'infidelity_occurred': 0,
        'counseling_attended': 0,
        'social_support': 7.0,
        'shared_hobbies_count': 3,
        'marriage_type': 'Love',
        'pre_marital_cohabitation': 1,
        'domestic_violence_history': 0,
        'trust_score': 8.0
    }
    
    try:
        prediction, probabilities = predictor.predict_new_sample(sample_person)
        print(f"Prediction for sample person: {'Divorced' if prediction == 1 else 'Not Divorced'}")
        print(f"Probability of divorce: {probabilities[1]:.2%}")
        print(f"Probability of staying married: {probabilities[0]:.2%}")
    except Exception as e:
        print(f"Error in prediction: {e}")
        print("This might be due to categorical encoding issues with new data.")


if __name__ == "__main__":
    main()