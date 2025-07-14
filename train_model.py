import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
df = pd.read_csv('adult 3.csv')

# Replace '?' with NaN and drop missing rows
df = df.replace('?', pd.NA).dropna()

# Features (X) and target (y)
X = df.drop('income', axis=1)
y = df['income']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)  # <=50K => 0, >50K => 1

# Separate categorical & numeric columns
categorical = X.select_dtypes(include='object').columns.tolist()
numerical = X.select_dtypes(include='int64').columns.tolist()

# Preprocessor: OneHot encode categorical columns
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Build pipeline: preprocessor + classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Save model & label encoder
joblib.dump(pipeline, 'income_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Model trained & saved!")
