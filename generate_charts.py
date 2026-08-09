import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# 1. Main Comparison Chart (Accuracy & F1)
# --------------------------------------------------

settings = [
    "Local\nClient 1\n(Express)",
    "Local\nClient 2\n(Standard)",
    "Federated\nLogistic Reg",
    "Federated\nNeural Net",
    "Centralized\nNeural Net"
]

accuracy = [0.8348, 0.6552, 0.5508, 0.6934, 0.6968]
f1_score = [0.9041, 0.5207, 0.6970, 0.6017, 0.6600]

x = np.arange(len(settings))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#4C72B0')
bars2 = ax.bar(x + width/2, f1_score, width, label='F1 Score', color='#55A868')

ax.set_ylabel('Score')
ax.set_title('Performance Comparison under Strong Non-IID (Shipping Mode Partition)')
ax.set_xticks(x)
ax.set_xticklabels(settings)
ax.legend()
ax.set_ylim(0, 1.05)
ax.bar_label(bars1, fmt='%.3f', padding=3, fontsize=8)
ax.bar_label(bars2, fmt='%.3f', padding=3, fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('figures/comparison_accuracy_f1.png', dpi=300)
plt.show()
print("Saved: figures/comparison_accuracy_f1.png")


# --------------------------------------------------
# 2. Learning Curves – Neural Network (5 Rounds)
# --------------------------------------------------

rounds = [1, 2, 3, 4, 5]
loss = [0.3365, 0.3088, 0.3064, 0.3051, 0.3062]
acc = [0.6635, 0.6912, 0.6936, 0.6949, 0.6938]
f1 = [0.6392, 0.6015, 0.6018, 0.6006, 0.6021]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss
axes[0].plot(rounds, loss, marker='o', linewidth=2, color='#C44E52')
axes[0].set_title('Loss over Rounds')
axes[0].set_xlabel('Round')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].set_xticks(rounds)

# Accuracy
axes[1].plot(rounds, acc, marker='o', linewidth=2, color='#4C72B0')
axes[1].set_title('Accuracy over Rounds')
axes[1].set_xlabel('Round')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].set_xticks(rounds)
axes[1].set_ylim(0.65, 0.71)

# F1 Score
axes[2].plot(rounds, f1, marker='o', linewidth=2, color='#55A868')
axes[2].set_title('F1 Score over Rounds')
axes[2].set_xlabel('Round')
axes[2].set_ylabel('F1 Score')
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].set_xticks(rounds)
axes[2].set_ylim(0.58, 0.66)

plt.suptitle('Neural Network Learning Curves (Strong Non-IID, 5 Rounds)', fontsize=14)
plt.tight_layout()
plt.savefig('figures/nn_learning_curves.png', dpi=300)
plt.show()
print("Saved: figures/nn_learning_curves.png")

print("\nAll charts generated successfully.")