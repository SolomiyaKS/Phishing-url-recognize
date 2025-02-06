import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler
import joblib

# Завантаження набору даних з parquet
splits = {'train': 'data/train-00000-of-00001-29d9936006e8a9c0.parquet', 
          'validation': 'data/validation-00000-of-00001-e38efd4eca4f515a.parquet', 
          'test': 'data/test-00000-of-00001-978966d811841a2e.parquet'}

# Завантаження тренувального датасету
df = pd.read_parquet("hf://datasets/Razvan27/remla_phishing_url/" + splits["train"])

# Перевірка наявності даних
print(f"Dataset loaded with {len(df)} samples")

# Розподіл на тренувальний і тестовий набори (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Підготовка даних для векторизації
X_train, y_train = train_df['url'], train_df['label']
X_test, y_test = test_df['url'], test_df['label']

# Векторизація URL (ngram_range=(1, 2) означає одиничні та двійні n-грамми)
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Масштабування ознак (використовуємо MaxAbsScaler для sparse матриць)
scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train_vec)  # Масштабування тренувальних даних
X_test_scaled = scaler.transform(X_test_vec)  # Масштабування тестових даних

# Навчання моделі (Logistic Regression)
model = LogisticRegression(max_iter=1000)  # Збільшення кількості ітерацій
model.fit(X_train_scaled, y_train)

# Тестування моделі
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Збереження моделі, векторизатора і масштабувальника
joblib.dump(model, "phishing_model2.pkl")
joblib.dump(vectorizer, "vectorizer2.pkl")
joblib.dump(scaler, "scaler2.pkl")

print("Model, vectorizer, and scaler saved successfully.")
