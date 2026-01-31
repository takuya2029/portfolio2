import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib

# =========================
# 設定
# =========================
SVM_THRESHOLD = 0.62
LOGI_THRESHOLD = 0.5

st.set_page_config(page_title="固化予測アプリ", layout="centered")
st.title("🧪 固化予測アプリ")

# =========================
# 日本語フォント対応
# =========================
matplotlib.rcParams["font.family"] = "Yu Gothic"

# =========================
# モデル・スケーラー読み込み
# =========================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("logistic_model.pkl", "rb") as f:
    logi = pickle.load(f)
with open("svm_model.pkl", "rb") as f:
    svm = pickle.load(f)
with open("feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# =========================
# 学習データ（標準化後・評価用）
# =========================
file_path_scaled = "ハイソリP_PAW_PWH_final_MI_r3 - コピー - コピー.csv"
df_scaled = pd.read_csv(file_path_scaled, encoding="utf-8")  # UTF-8
target_col = "固化"
# モデル評価用データ（標準化前CSVを使用）
file_path_eval = "ハイソリP_PAW_PWH_final_MI_r3 _大元.csv"
df_eval = pd.read_csv(file_path_eval, encoding="utf-8")
X_eval = df_eval[feature_cols].select_dtypes(include=[float, int])
y_eval = df_eval[target_col]

# 標準化
X_eval_scaled = scaler.transform(X_eval)



# =========================
# 標準化前データ（代表値用）
# =========================
file_path_orig = "ハイソリP_PAW_PWH_final_MI_r3 _大元.csv"
df_orig = pd.read_csv(file_path_orig, encoding="utf-8")
X_orig = df_orig[feature_cols].select_dtypes(include=[float, int])
X_mean_orig = X_orig.mean()  # 標準化前の平均値

# =========================
# タブ作成
# =========================
tab1, tab2 = st.tabs(["固化予測", "モデル評価"])

# =========================
# タブ1: 固化予測
# =========================
with tab1:
    st.subheader("特徴量入力で固化予測")

    # 「代表値にリセット」ボタン
    if st.button("入力値を代表値にリセット", key="reset_mean"):
        input_values = X_mean_orig.to_dict()
    else:
        input_values = {col: st.number_input(col, value=X_mean_orig[col]) for col in feature_cols}

    input_df = pd.DataFrame([input_values])

    # モデル入力用に標準化
    X_input_scaled = scaler.transform(input_df[feature_cols])

    # モデル選択
    model_type = st.radio(
        "解析モデルを選択してください",
        ["ロジスティック回帰", "SVM（RBF）"]
    )

    if st.button("予測", key="predict"):
        if model_type == "ロジスティック回帰":
            proba = logi.predict_proba(X_input_scaled)[0, 1]
            threshold = LOGI_THRESHOLD
        else:
            proba = svm.predict_proba(X_input_scaled)[0, 1]
            threshold = SVM_THRESHOLD

        pred = int(proba >= threshold)

        st.metric("固化確率", f"{proba:.3f}")
        if pred == 1:
            st.error("固化と予測")
        else:
            st.success("非固化と予測")
        st.caption(f"判定閾値: {threshold}")

# =========================
# タブ2: モデル評価
# =========================
with tab2:
    st.subheader("モデル精度評価（学習データ）")
    model_type_eval = st.radio(
        "評価するモデルを選択してください",
        ["ロジスティック回帰", "SVM（RBF）"]
    )

    if st.button("評価実行", key="eval"):
        if model_type_eval == "ロジスティック回帰":
            y_pred = (logi.predict_proba(X_eval_scaled)[:, 1] >= LOGI_THRESHOLD).astype(int)
        else:
            y_pred = (svm.predict_proba(X_eval_scaled)[:, 1] >= SVM_THRESHOLD).astype(int)


        # 精度表示
        acc = accuracy_score(y_eval, y_pred)
        st.write(f"✅ 正解率 (Accuracy): {acc:.3f}")

        # 混同行列
        cm = confusion_matrix(y_eval, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("予測")
        ax.set_ylabel("実際")
        st.pyplot(fig)

        # 詳細レポート
        st.text(classification_report(y_eval, y_pred, digits=3))


        # ロジスティック回帰のみ特徴量重要度
        if model_type_eval == "ロジスティック回帰":
            st.subheader("特徴量重要度（係数）")
            coef_df = pd.DataFrame({
                "特徴量": feature_cols,
                "重要度": logi.coef_[0]
            }).sort_values(by="重要度", key=abs, ascending=False)

            fig2, ax2 = plt.subplots(figsize=(8, max(4, len(feature_cols)*0.3)))
            sns.barplot(x="重要度", y="特徴量", data=coef_df, ax=ax2, palette="viridis")
            ax2.set_title("ロジスティック回帰係数")
            st.pyplot(fig2)


