import json

# Path to your downloaded notebook
in_file = "AgriVerse_AllInOne_Trainer.ipynb"
out_file = "AgriVerse_AllInOne_Trainer_fixed.ipynb"

with open(in_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove 'widgets' metadata if present and broken
if "widgets" in nb.get("metadata", {}):
    nb["metadata"].pop("widgets", None)

# Also clean per-cell metadata if needed
for cell in nb.get("cells", []):
    if "metadata" in cell and "widgets" in cell["metadata"]:
        cell["metadata"].pop("widgets", None)

# Save cleaned file
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Fixed notebook saved to {out_file}")
