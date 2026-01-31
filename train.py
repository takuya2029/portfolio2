import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# =========================
# データ読み込み
# =========================
file_path = r"C:\Users\mt100\Downloads\ポートフォリオ作成\固化予測アプリ\ハイソリP_PAW_PWH_final_MI_r3 _大元.csv"
df = pd.read_csv(file_path, encoding="utf-8")

target_col = "固化"

X = df.drop(columns=[target_col])
X = X.select_dtypes(include=[np.number])
y = df[target_col]

# =========================
# スケーリング
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# Logistic Regression
# =========================
logi = LogisticRegression(max_iter=1000, random_state=42)
logi.fit(X_scaled, y)

# =========================
# SVM
# =========================
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)
svm.fit(X_scaled, y)

# =========================
# 保存
# =========================
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("logistic_model.pkl", "wb") as f:
    pickle.dump(logi, f)

with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ 学習＆保存完了")
