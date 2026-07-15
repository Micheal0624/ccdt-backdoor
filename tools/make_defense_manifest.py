import csv
import os

rows = []

datasets = ["cifar10", "gtsrb", "cifar100"]
poisons = [0.05, 0.01]
defenses = ["strip", "neural_cleanse", "spectral_signatures", "fine_pruning"]

# ResNet-18 主矩阵：naive_dual + full
for dataset in datasets:
    for poison in poisons:
        for method in ["naive_dual", "full"]:
            for defense in defenses:
                rows.append({
                    "dataset": dataset,
                    "model": "resnet18",
                    "method": method,
                    "poison_rate": poison,
                    "seed": 0,
                    "kc": 2,
                    "position_mode": "dynamic",
                    "defense": defense,
                })

# VGG11 架构补充：full only
for dataset in datasets:
    for poison in poisons:
        for defense in defenses:
            rows.append({
                "dataset": dataset,
                "model": "vgg11",
                "method": "full",
                "poison_rate": poison,
                "seed": 0,
                "kc": 2,
                "position_mode": "dynamic",
                "defense": defense,
            })

os.makedirs("results/tables", exist_ok=True)
out = "results/tables/defense_manifest_seed0.csv"

with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", out)
print("Rows:", len(rows))

for r in rows[:20]:
    print(r)
