from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def print_and_store_results(name, y_true, y_pred, results):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    results[name] = acc
    return acc

def plot_results(results):
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=["skyblue", "salmon", "gold", "lightgreen"])
    plt.ylabel("Accuracy")
    plt.title("Model Comparison - Financial Sentiment")
    plt.ylim(0, 1)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
