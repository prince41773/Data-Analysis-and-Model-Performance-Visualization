import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def train_and_evaluate_models(X_train, X_test, y_train, y_test, target_type):
    results = {}
    confusion_matrices = {}
    models = []

    if target_type == "classification":
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=500)),
            ("Random Forest", RandomForestClassifier(n_estimators=100)),
            ("Support Vector Machine", SVC()),
        ]
        metric = "accuracy"
    else:  # regression
        models = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest Regressor", RandomForestRegressor(n_estimators=100)),
            ("Support Vector Regressor", SVR()),
        ]
        metric = "r2_score"

    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if target_type == "classification":
            score = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)
            cm_path = os.path.join("static", f"{name.replace(' ', '_')}_confusion_matrix.png")
            plot_confusion_matrix(cm, name, cm_path)
            confusion_matrices[name] = cm_path
        else:  # regression
            score = r2_score(y_test, predictions)

        results[name] = {metric: round(score, 4)}

    comparison_chart = plot_model_comparison(results, target_type)
    return results, confusion_matrices, comparison_chart


def plot_confusion_matrix(cm, model_name, save_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_model_comparison(results, target_type):
    metric = "accuracy" if target_type == "classification" else "r2_score"
    model_names = list(results.keys())
    scores = [results[model][metric] for model in model_names]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=scores, palette='viridis')
    plt.title(f'Model Comparison ({metric.capitalize()})')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45, ha='right')
    chart_path = os.path.join("static", "model_comparison.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path
