import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the CSV file
df = pd.read_csv("prediction_results_PSC_Liver_new_splitting_technique.csv")

# Extract true and predicted labels
y_true = df['mapped_cell_type']
# y_pred = df['sctype_classification']
y_pred = df['Pred_labels']

# Calculate accuracy
acc = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy: {acc:.4f}")

# Detailed classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

# Optional: Display confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred, labels=y_true.unique())
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=y_true.unique(), yticklabels=y_true.unique(), cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
