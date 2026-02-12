# -*- coding: utf-8 -*-
"""
Shows how the model did in detail: metrics and plots you can save.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

plt.rcParams["figure.figsize"] = (8, 5)


def metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print("F1:", round(f1_score(y_true, y_pred), 4))


def confusion_plot(y_true, y_pred, save=True):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("What the model said")
    plt.ylabel("What actually happened")
    plt.title("Confusion matrix")
    plt.tight_layout()
    if save:
        base = Path(__file__).resolve().parent.parent
        (base / "reports").mkdir(exist_ok=True)
        plt.savefig(base / "reports" / "confusion_matrix.png", dpi=150)
        print("Plot saved to reports/confusion_matrix.png")
    plt.show()
