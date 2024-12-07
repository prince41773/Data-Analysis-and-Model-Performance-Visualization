import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

def process_data(filepath):
    data = pd.read_csv(filepath)

    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)
    data.fillna("Unknown", inplace=True)

    # Detect target variable (assume last column)
    target_col = data.columns[-1]
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    # Determine if it's classification or regression
    target_type = "classification" if data[target_col].nunique() < 15 else "regression"

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Data description
    data_summary = data.describe(include='all').transpose().to_dict()

    # Generate heatmap
    os.makedirs('static', exist_ok=True)
    heatmap_path = 'static/heatmap.png'
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    return data_summary, heatmap_path, X_train, X_test, y_train, y_test, target_type

def perform_eda(filepath):
    data = pd.read_csv(filepath)
    plots = []
    save_dir = 'static/eda/'
    os.makedirs(save_dir, exist_ok=True)

    # Histograms
    for column in data.select_dtypes(include=['int64', 'float64']).columns:
        plot_path = os.path.join(save_dir, f"{column}_hist.png")
        sns.histplot(data[column], kde=True, color='blue')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plots.append(plot_path)

    # Boxplots for numeric columns
    for column in data.select_dtypes(include=['int64', 'float64']).columns:
        plot_path = os.path.join(save_dir, f"{column}_boxplot.png")
        sns.boxplot(x=data[column], color='orange')
        plt.title(f"Boxplot of {column}")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        plots.append(plot_path)

    # Pair plot (if applicable)
    if len(data.columns) <= 5:
        pair_plot_path = os.path.join(save_dir, "pair_plot.png")
        sns.pairplot(data)
        plt.tight_layout()
        plt.savefig(pair_plot_path)
        plt.close()
        plots.append(pair_plot_path)

    # Count plots for categorical columns
    if data.select_dtypes(include='object').shape[1] > 0:
        for col in data.select_dtypes(include='object').columns:
            count_plot_path = os.path.join(save_dir, f"{col}_count.png")
            sns.countplot(y=data[col], palette="viridis")
            plt.title(f"Count Plot of {col}")
            plt.tight_layout()
            plt.savefig(count_plot_path)
            plt.close()
            plots.append(count_plot_path)

    return plots
