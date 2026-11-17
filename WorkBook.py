# =====================================================
# INSTALACAO (COLAB)
# =====================================================

# !pip install xgboost shap -q

# =====================================================
# IMPORTACOES
# =====================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import shap

# =====================================================
# LEITURA DOS DADOS
# =====================================================

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

print(df.head())

# =====================================================
# ANALISE DO DESBALANCEAMENTO
# =====================================================

print("\nDistribuicao das classes:")
print(df["Class"].value_counts())

print("\nDistribuicao percentual:")
print(df["Class"].value_counts(normalize=True))

percentuais = (
    df["Class"]
    .value_counts(normalize=True)
    * 100
)

plt.figure(figsize=(6,4))

sns.barplot(
    x=percentuais.index,
    y=percentuais.values
)

plt.title("Distribuição Percentual das Classes")
plt.xlabel("Classe")
plt.ylabel("Percentual (%)")

for i, v in enumerate(percentuais.values):
    plt.text(i, v + 0.1, f"{v:.2f}%")

plt.show()

# =====================================================
# PREPARACAO DOS DADOS
# =====================================================

df["LogAmount"] = np.log1p(df["Amount"])

X = df.drop(
    columns=["Class"]
)

y = df["Class"]

# =====================================================
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# MODELO 1 - LOGISTIC REGRESSION
# =====================================================

log_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000
            )
        )
    ]
)

log_model.fit(
    X_train,
    y_train
)

pred_log = log_model.predict(
    X_test
)

print("\n===== LOGISTIC REGRESSION =====")

print(
    classification_report(
        y_test,
        pred_log
    )
)

# =====================================================
# MODELO 2 - RANDOM FOREST
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

pred_rf = rf_model.predict(
    X_test
)

print("\n===== RANDOM FOREST =====")

print(
    classification_report(
        y_test,
        pred_rf
    )
)

# =====================================================
# MODELO 3 - XGBOOST
# =====================================================

fraudes = y_train.sum()

normais = len(y_train) - fraudes

scale_pos_weight = normais / fraudes

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

xgb_model.fit(
    X_train,
    y_train
)

pred_xgb = xgb_model.predict(
    X_test
)

print("\n===== XGBOOST =====")

print(
    classification_report(
        y_test,
        pred_xgb
    )
)

# =====================================================
# GRID SEARCH
# =====================================================

param_grid = {
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.05],
    "n_estimators": [200, 300]
}

grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="recall",
    cv=3,
    n_jobs=-1
)

grid.fit(
    X_train,
    y_train
)

print("\nMelhores parametros:")
print(grid.best_params_)

best_model = grid.best_estimator_

# =====================================================
# THRESHOLD 0.30
# =====================================================

probs = best_model.predict_proba(
    X_test
)[:, 1]

threshold = 0.30

pred_threshold = (
    probs >= threshold
).astype(int)

print("\n===== XGBOOST COM THRESHOLD 0.30 =====")

print(
    classification_report(
        y_test,
        pred_threshold
    )
)

# =====================================================
# ROC CURVE
# =====================================================

fpr, tpr, _ = roc_curve(
    y_test,
    probs
)

plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr
)

plt.title(
    "ROC Curve"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.grid()

plt.show()

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        probs
    )
)

# =====================================================
# PRECISION RECALL CURVE
# =====================================================

precision, recall, _ = precision_recall_curve(
    y_test,
    probs
)

plt.figure(figsize=(8, 5))

plt.plot(
    recall,
    precision
)

plt.title(
    "Precision Recall Curve"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.grid()

plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importancias = pd.Series(
    best_model.feature_importances_,
    index=X.columns
)

top10 = (
    importancias
    .sort_values(
        ascending=False
    )
    .head(10)
)

print("\nTop 10 Features")
print(top10)

top10.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title(
    "Top 10 Features"
)

plt.show()

# =====================================================
# SHAP - GRAFICO DE BARRAS
# =====================================================

explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    max_display=10,
    show=False
)

plt.title("Top 10 Variáveis Mais Influentes (SHAP)")
plt.tight_layout()
plt.show()