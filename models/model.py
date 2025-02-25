import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler
import joblib

# Loading a dataset from parquet
splits = {'train': 'data/train-00000-of-00001-29d9936006e8a9c0.parquet', 
          'validation': 'data/validation-00000-of-00001-e38efd4eca4f515a.parquet', 
          'test': 'data/test-00000-of-00001-978966d811841a2e.parquet'}

# Loading the training dataset
df = pd.read_parquet("hf://datasets/Razvan27/remla_phishing_url/" + splits["train"])

# Check the availability of data
print(f"Dataset loaded with {len(df)} samples")

# Split into training and test sets (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Preparing data for vectorisation
X_train, y_train = train_df['url'], train_df['label']
X_test, y_test = test_df['url'], test_df['label']

# URL vectorisation (ngram_range=(1, 2) means single and double n-grams)
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Feature scaling (use MaxAbsScaler for sparse matrices)
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train_vec) 
X_test_scaled = scaler.transform(X_test_vec)

# Model training (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Model testing
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Saving the model, vectoriser, and scaler
joblib.dump(model, "phishing_model2.pkl")
joblib.dump(vectorizer, "vectorizer2.pkl")
joblib.dump(scaler, "scaler2.pkl")

print("Model, vectorizer, and scaler saved successfully.")
