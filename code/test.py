"""
MAI/IDL SS26 - Final assignment.

Simple test/evaluation script.
Loads the best checkpoint for the selected DATA and MODEL from config.json,
evaluates the test set, prints required metrics, and appends them to a results file.
"""

import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data import get_loaders
import models


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing executing on device: {device}")

    _, _, test_loader = get_loaders(
        data=config["DATA"],
        data_path=config["DATA_PATH"],
        batch_size=config["BATCH_SIZE"]
    )

    model_class = getattr(models, config["MODEL"])
    model = model_class(
        in_channels=config["CHANNELS"],
        num_classes=config["NUM_CLASSES"]
    ).to(device)

    checkpoint_path = Path(f"checkpoints/{config['DATA']}_{config['MODEL']}.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_labels = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    inference_time = time.time() - start_time
    latency_ms_per_sample = (inference_time / len(all_labels)) * 1000

    test_acc = accuracy_score(all_labels, all_predictions) * 100
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0) * 100
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0) * 100
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0) * 100

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("-" * 50)
    print(f"Dataset: {config['DATA']}")
    print(f"Model: {config['MODEL']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Best Epoch: {checkpoint.get('best_epoch', 'unknown')}")
    print(f"Best Train Loss: {checkpoint.get('train_loss', 0):.4f}")
    print(f"Best Train Acc: {checkpoint.get('train_acc', 0):.2f}%")
    print(f"Best Val Loss: {checkpoint.get('val_loss', 0):.4f}")
    print(f"Best Val Acc: {checkpoint.get('val_acc', 0):.2f}%")
    print(f"Test Acc: {test_acc:.2f}%")
    print(f"Macro Precision: {precision:.2f}%")
    print(f"Macro Recall: {recall:.2f}%")
    print(f"Macro F1: {macro_f1:.2f}%")
    print(f"Inference Latency: {latency_ms_per_sample:.4f} ms/sample")
    print(f"Peak Memory: {peak_memory_mb:.2f} MB")
    print("-" * 50)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "benchmark_results.txt"

    if not results_file.exists():
        results_file.write_text(
            "DATA\tMODEL\tCHANNELS\tNUM_CLASSES\tLR\tEPOCHS\tBATCH_SIZE\t"
            "BEST_EPOCH\tBEST_TRAIN_LOSS\tBEST_TRAIN_ACC\tBEST_VAL_LOSS\tBEST_VAL_ACC\t"
            "TEST_ACC\tMACRO_PRECISION\tMACRO_RECALL\tMACRO_F1\t"
            "INFERENCE_MS_PER_SAMPLE\tPEAK_MEMORY_MB\tCHECKPOINT\n"
        )

    with results_file.open("a") as f:
        f.write(
            f"{config['DATA']}\t"
            f"{config['MODEL']}\t"
            f"{config['CHANNELS']}\t"
            f"{config['NUM_CLASSES']}\t"
            f"{config['LEARNING_RATE']}\t"
            f"{config['EPOCHS']}\t"
            f"{config['BATCH_SIZE']}\t"
            f"{checkpoint.get('best_epoch', '')}\t"
            f"{checkpoint.get('train_loss', 0):.4f}\t"
            f"{checkpoint.get('train_acc', 0):.2f}\t"
            f"{checkpoint.get('val_loss', 0):.4f}\t"
            f"{checkpoint.get('val_acc', 0):.2f}\t"
            f"{test_acc:.2f}\t"
            f"{precision:.2f}\t"
            f"{recall:.2f}\t"
            f"{macro_f1:.2f}\t"
            f"{latency_ms_per_sample:.4f}\t"
            f"{peak_memory_mb:.2f}\t"
            f"{checkpoint_path}\n"
        )

    print(f"Appended results to: {results_file}")


if __name__ == "__main__":
    main()