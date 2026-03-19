"""Demo: classify temporal graph evolution patterns.

We generate synthetic graphs under three evolution patterns:
  - "random"    (edges form/dissolve randomly)
  - "community" (strong cluster structure)
  - "growing"   (preferential-attachment growth)

A DynamicGraphDiffusionNet is trained to distinguish between them.
This exercises the full pipeline end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

from dgdn import DynamicGraphDiffusionNet, SyntheticTemporalGraph


def make_dataset(n_per_class=40, num_nodes=15, num_steps=6, feature_dim=8, seed=0):
    """Generate labelled training examples."""
    patterns = ["random", "community", "growing"]
    label_map = {p: i for i, p in enumerate(patterns)}
    data = []
    for cls_idx, pat in enumerate(patterns):
        for i in range(n_per_class):
            gen = SyntheticTemporalGraph(
                num_nodes=num_nodes,
                num_steps=num_steps,
                feature_dim=feature_dim,
                pattern=pat,
                seed=seed + cls_idx * 1000 + i,
            )
            snaps = gen.generate()
            data.append((snaps, cls_idx))
    return data


def train(model, data, epochs=60, lr=1e-3, weight_decay=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    indices = list(range(len(data)))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0

        # Shuffle each epoch
        torch.manual_seed(epoch)
        perm = torch.randperm(len(indices)).tolist()

        for i in perm:
            snaps, label = data[i]
            optimizer.zero_grad()
            logits = model(snaps)              # [num_classes]
            target = torch.tensor([label])
            loss = loss_fn(logits.unsqueeze(0), target)
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += int(logits.argmax().item() == label)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            acc = correct / len(data) * 100
            print(f"Epoch {epoch:3d}  loss={total_loss/len(data):.4f}  "
                  f"train_acc={acc:.1f}%")


def evaluate(model, data):
    model.eval()
    correct = 0
    with torch.no_grad():
        for snaps, label in data:
            logits = model(snaps)
            correct += int(logits.argmax().item() == label)
    return correct / len(data)


def main():
    print("=== Dynamic Graph Diffusion Network Demo ===\n")
    print("Task: classify temporal graph evolution patterns")
    print("  Class 0: random    — edges form/dissolve uniformly")
    print("  Class 1: community — strong cluster structure")
    print("  Class 2: growing   — preferential-attachment growth\n")

    torch.manual_seed(42)

    # Build dataset
    train_data = make_dataset(n_per_class=40, seed=0)
    test_data  = make_dataset(n_per_class=20, seed=9999)

    print(f"Train: {len(train_data)} graphs  |  Test: {len(test_data)} graphs\n")

    # Build model
    model = DynamicGraphDiffusionNet(
        in_features=8,
        hidden_dim=64,
        num_diff_layers=2,
        diffusion_t=1.5,
        num_diff_steps=5,
        num_attn_heads=4,
        time_dim=16,
        attn_pool="attn",
        dropout=0.1,
        num_classes=3,
        task="graph",
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # Train
    train(model, train_data, epochs=60, lr=1e-3)

    # Evaluate
    test_acc = evaluate(model, test_data)
    print(f"\nTest accuracy: {test_acc*100:.1f}%")

    # Show per-class breakdown
    patterns = ["random", "community", "growing"]
    print("\nPer-class test accuracy:")
    model.eval()
    with torch.no_grad():
        for cls_idx, pat in enumerate(patterns):
            cls_data = [(s, l) for s, l in test_data if l == cls_idx]
            c = sum(1 for s, l in cls_data if model(s).argmax().item() == l)
            print(f"  {pat:10s}: {c}/{len(cls_data)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
