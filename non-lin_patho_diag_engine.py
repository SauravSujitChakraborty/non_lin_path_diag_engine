import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# ==========================================
# 1. GENERATE CELLULAR DATA (Synthetic Biopsy)
# ==========================================
np.random.seed(42)
n_samples = 200

# Features: Radius, Texture, Perimeter, Smoothness
data = {
    'Radius': np.random.normal(14, 3, n_samples),
    'Texture': np.random.normal(19, 4, n_samples),
    'Perimeter': np.random.normal(92, 24, n_samples),
    'Smoothness': np.random.normal(0.1, 0.01, n_samples)
}
df = pd.DataFrame(data)

# Malignant Logic: Higher radius/texture = Malignant(1)
df['Target'] = ((df['Radius'] > 16) | (df['Texture'] > 22)).astype(int)

# ==========================================
# 2. PREPROCESSING & SCALING
# ==========================================
feature_cols = ['Radius', 'Texture', 'Perimeter', 'Smoothness']
X = df[feature_cols]
y = df['Target']

# SVM is scale-sensitive; StandardScaler transforms data to mean=0, std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y)

# ==========================================
# 3. TRAIN THE SVM CLASSIFIER (RBF Kernel)
# ==========================================
# probability=True allows us to use predict_proba for risk percentages
model = SVC(kernel='rbf', probability=True, C=1.0)
model.fit(X_train, y_train)

# ==========================================
# 4. DIAGNOSTIC INTERFACE (Cleaned & Corrected)
# ==========================================
def predict_biopsy(radius, texture, perimeter, smoothness):
    # Step A: Create a DataFrame with names to avoid the Pink Warning
    sample_raw = pd.DataFrame([[radius, texture, perimeter, smoothness]], 
                              columns=feature_cols)
    
    # Step B: Scale using the already-fitted scaler
    sample_scaled = scaler.transform(sample_raw)
    
    # Step C: Predict probability and class
    prob = model.predict_proba(sample_scaled)[0][1]
    prediction = model.predict(sample_scaled)[0]
    
    result = "🔴 MALIGNANT (High Concern)" if prediction == 1 else "🟢 BENIGN (Low Concern)"
    
    print(f"\n--- Pathology Lab Report ---")
    print(f"Measurements: Radius {radius}, Texture {texture}")
    print(f"Probability of Malignancy: {prob:.1%}")
    print(f"Final Assessment: {result}")

# --- EXECUTION & RESULTS ---
print(f"System Readiness: {accuracy_score(y_test, model.predict(X_test)):.2%} Accuracy")

# CASE 1: High-risk measurements
predict_biopsy(18.5, 25.0, 120.0, 0.12) 

# CASE 2: Low-risk measurements
predict_biopsy(11.2, 15.0, 70.0, 0.08)

