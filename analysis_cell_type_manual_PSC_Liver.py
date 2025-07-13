import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

# Load the CSV file
file_path = "output/sctype_annotations_PSC_Liver.csv"
df = pd.read_csv(file_path)

# Show all unique entries from 'cell_type'
print("Unique cell types:")
for cell_type in sorted(df['cell_type'].unique()):
    print("-", cell_type)

# Show all unique entries from 'sctype_classification'
print("\nUnique ScType predictions:")
for cell_type in sorted(df['sctype_classification'].unique()):
    print("-", cell_type)

# ----------------------------
# Step 2: Mapping dictionary
# ----------------------------
label_mapping = {
    'hepatocyte': 'Hepatocytes',
    'centrilobular region hepatocyte': 'Hepatocytes',
    'periportal region hepatocyte': 'Hepatocytes',
    'midzonal region hepatocyte': 'Hepatocytes',

    'endothelial cell of artery': 'Endothelial cell',
    'endothelial cell of pericentral hepatic sinusoid': 'Endothelial cell',
    'endothelial cell of periportal hepatic sinusoid': 'Endothelial cell',
    'vein endothelial cell': 'Endothelial cell',

    'Kupffer cell': 'Kupffer cells',

    'macrophage': 'Immune system cells',
    'monocyte': 'Immune system cells',
    'neutrophil': 'Immune system cells',
    'mast cell': 'Immune system cells',
    'T cell': 'Immune system cells',
    'CD4-positive, alpha-beta T cell': 'Immune system cells',
    'CD8-positive, alpha-beta T cell': 'Immune system cells',
    'natural killer cell': 'Immune system cells',
    'hepatic pit cell': 'Immune system cells',
    'conventional dendritic cell': 'Immune system cells',
    'plasmacytoid dendritic cell': 'Immune system cells',
    'mature B cell': 'Immune system cells',
    'plasma cell': 'Immune system cells',

    'hepatic stellate cell': 'Unknown',
    'intrahepatic cholangiocyte': 'Unknown',
    'fibroblast': 'Unknown',
    'erythrocyte': 'Unknown',
    'unknown': 'Unknown',
}

# ----------------------------
# Step 3: Apply mapping
# ----------------------------
df['mapped_cell_type'] = df['cell_type'].map(label_mapping)

# Save new CSV
output_file = "sctype_annotations_mapped_CELLTYPE_PSC_Liver.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Mapping complete. File saved as '{output_file}'")

# ----------------------------
# Step 4: Confusion Matrix + Metrics
# ----------------------------
y_true = df['mapped_cell_type']
y_pred = df['sctype_classification']

# Unified label set
labels = sorted(set(y_true) | set(y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix: Mapped True Labels vs ScType Predictions")
plt.xlabel("Predicted Label (ScType)")
plt.ylabel("True Label (Mapped from Paper)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
# plt.show()

# ----------------------------
# Step 5: Evaluation Metrics
# ----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\n🔍 Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Detailed Report
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
