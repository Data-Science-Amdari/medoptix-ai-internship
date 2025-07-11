
# utils.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    """Display a seaborn confusion matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances, feature_names, top_n=10):
    """Plot top N feature importances."""
    sorted_idx = importances.argsort()[-top_n:]
    plt.figure(figsize=(8, 5))
    plt.barh(range(top_n), importances[sorted_idx])
    plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()
