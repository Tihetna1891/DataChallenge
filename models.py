import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# Load the dataset
df = pd.read_csv('C:/Users/dell/Downloads/CHAMPS.csv')

# Display the number of rows and columns
rows, columns = df.shape
rows, columns
# Rename the columns
df.rename(columns={
    'champs_id': 'CHAMPS_ID',
    'dp_013': 'Case_Type',
    'dp_108': 'Underlying_Cause',
    'dp_118': 'Maternal_Condition'
}, inplace=True)

# Rename specific values in the 'Case_Type' column
df['Case_Type'].replace({
    'CH00716': 'Stillbirth',
    'CH01404': 'Death in the first 24 hours',
    'CH01405': 'Early Neonate (1 to 6 days)',
    'CH01406': 'Late Neonate (7 to 27 days)',
    'CH00718': 'Infant (28 days to less than 12 months)',
    'CH00719': 'Child (12 months to less than 60 months)'
}, inplace=True)

# Show the proportion of null values in each column
null_proportions = df.isnull().mean()
null_proportions
# Magnitude and proportion of each underlying cause of child death
underlying_cause_counts = df['Underlying_Cause'].value_counts()
underlying_cause_proportions = df['Underlying_Cause'].value_counts(normalize=True)

# Magnitude and proportion of maternal factors
maternal_condition_counts = df['Maternal_Condition'].value_counts()
maternal_condition_proportions = df['Maternal_Condition'].value_counts(normalize=True)

# Proportion of child death by case type
case_type_counts = df['Case_Type'].value_counts()
case_type_proportions = df['Case_Type'].value_counts(normalize=True)

underlying_cause_counts, underlying_cause_proportions, maternal_condition_counts, maternal_condition_proportions, case_type_counts, case_type_proportions

# Encode categorical variables
label_encoder = LabelEncoder()

df['Case_Type_Encoded'] = label_encoder.fit_transform(df['Case_Type'])
df['Underlying_Cause_Encoded'] = label_encoder.fit_transform(df['Underlying_Cause'])
df['Maternal_Condition_Encoded'] = label_encoder.fit_transform(df['Maternal_Condition'])

# Define features and target variable for model training
X = df[['Case_Type_Encoded', 'Maternal_Condition_Encoded']]
y = df['Underlying_Cause_Encoded']

# Binarize the output
y_binarized = label_binarize(y, classes=np.unique(y))
# Define features and target variable for model training
X = df[['Case_Type_Encoded', 'Maternal_Condition_Encoded']]
y = df['Underlying_Cause_Encoded']

# Binarize the output
y_binarized = label_binarize(y, classes=np.unique(y))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_binarized = label_binarize(y_train, classes=np.unique(y))
y_test_binarized = label_binarize(y_test, classes=np.unique(y))

# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'AdaBoost': AdaBoostClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier()
}

model_results = {}
for model_name, model in models.items():
    ovr_model = OneVsRestClassifier(model)
    ovr_model.fit(X_train, y_train_binarized)
    y_pred = ovr_model.predict(X_test)
    y_prob = ovr_model.predict_proba(X_test)
    accuracy = accuracy_score(y_test_binarized, y_pred)
    # auc = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr')
    model_results[model_name] = {'accuracy': accuracy}

# Print model results
for model_name, metrics in model_results.items():
    print(f"{model_name}: Accuracy = {metrics['accuracy']:.2f}")

# Plot ROC curves for each model
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    ovr_model = OneVsRestClassifier(model)
    ovr_model.fit(X_train, y_train_binarized)
    y_prob = ovr_model.predict_proba(X_test)
    
    for i in range(y_prob.shape[1]):
        RocCurveDisplay.from_predictions(y_test_binarized[:, i], y_prob[:, i], name=f"{model_name} (class {i})")
    
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# Feature importance for models that provide it
for model_name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        ovr_model = OneVsRestClassifier(model).fit(X_train, y_train_binarized)
        plt.barh(X.columns, ovr_model.estimators_[0].feature_importances_)
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance for {model_name}')
        plt.show()

# Plot the top five infant underlying causes of child death
top5_infant_causes = underlying_cause_counts.head(5)
top5_infant_causes.plot(kind='bar', title='Top 5 Infant Underlying Causes of Child Death')
plt.ylabel('Count')
plt.show()

# Plot the top five maternal factors contributing to child death
top5_maternal_factors = maternal_condition_counts.head(5)
top5_maternal_factors.plot(kind='bar', title='Top 5 Maternal Factors Contributing to Child Death')
plt.ylabel('Count')
plt.show()

# Plot the child death based on case types
case_type_counts.plot(kind='bar', title='Child Death by Case Types')
plt.ylabel('Count')
plt.show()