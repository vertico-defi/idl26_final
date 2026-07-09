"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training executing on device: {device}")

    train_loader, val_loader, _ = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=0.99, activation_str=None).to(device)
    # (fix2) for the following crossEntropyLoss function there is an error saying the input needs to be a tensor. There are no inputs here, let's
    # (fix2) trace criterion. 
    # (fix3) now the second parameter in the cross entropy loss giving us an issue, no need to trace this time
    # (fix3) go straight to fit.py and see what the dimensions for labels are.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    # (fix2) criterion gets passed in here, the second argument to Trainer which comes from the fit.py file
    trainer = Trainer(model, criterion, optimizer, device)
    # (fix3) here is the train_loader in the method from fit.py, we see that it is an output from
    # (fix3) get_loaders which is the main class in data.py, so go there.
    checkpoint_path = f"checkpoints/{config['DATA']}_{config['MODEL']}.pt"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_start_time = time.time()

    best_metrics = trainer.fit(
        train_loader,
        val_loader,
        epochs=config["EPOCHS"],
        checkpoint_path=checkpoint_path
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_runtime_seconds = time.time() - train_start_time

    training_peak_memory_mb = 0.0
    if device.type == "cuda":
        training_peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Training Runtime: {training_runtime_seconds:.2f} seconds")
    print(f"Training Peak Memory: {training_peak_memory_mb:.2f} MB")
    print(f"Saved best model to: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint["training_runtime_seconds"] = training_runtime_seconds
    checkpoint["training_peak_memory_mb"] = training_peak_memory_mb
    torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    main()