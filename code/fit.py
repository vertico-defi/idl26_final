"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path

# (fix2) Here we see the criterion as an input to trainer
class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
    # the train_one_epoch method is called in the fit method
    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, sum = 0, 0
        
        # (fix3) dataloaders is coming in as a parameter to train one epoch.
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            # (fix4) Add the zero grad function to zero the gradients after each pass
            self.optimizer.zero_grad()
            # (fix2) outputs is assigned from the self.model(images) call. The model is pulled from 
            # (fix2) resnet18 class as setup in the config.json file. So the next line is calling
            # (fix2) the REsNEt18 object with images as the argument. Self.model(images) makes the model
            # (fix2) object callable. PyTorch internally routes that call to ResNet18.forward(self. images)
            outputs = self.model(images)
            # (fix2) SANITY CHECK: This self.criterion is nn.CrossEntropyLoss() fromt train.py
            # (fix2) so these inputs can be viewed as inputs into the nn.CrossEntropyLoss() object
            # (fix2) and compares the predicted outputs to the labels. Check the outputs size here
            # print("outputs type:", type(outputs))
            # print("outputs calue:", outputs)
            # (fix3) The error here is now the second parameter for the cross entropy loss object
            # (fix3) so here we'll just print the info on labels and see what is going on
            # print("labels type:", type(labels))
            # print("labels shape:", labels.shape)
            # print("labels dtype:", labels.dtype)
            # print("labels value:", labels)
            # (fix3) the hape of the labels is wrong, we'll need to trace it back, it is coming from dataloaders.
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            sum += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / sum, (correct / sum) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return running_loss / total, (correct / total) * 100

    def fit(self, train_loader, val_loader, epochs, checkpoint_path=None):
        print("\n Starting Training Routine...")
        print("-" * 50)

        best_val_acc = 0.0
        best_metrics = None
        
        for epoch in range(epochs):
            # (fix3) we can see here that the labels will come from train_loader.
            # (fix3) that is a parameter for the fit method. Thge fit method is called
            # (fix3) under the Trainer class in train.py
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
                
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = {
                    "best_epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }

                if checkpoint_path is not None:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "best_epoch": epoch + 1,
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc
                        },
                        checkpoint_path
                    )

            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        print("-" * 50)
        print("Training Complete!")

        if best_metrics is not None:
            print(f"Best Val Acc: {best_metrics['val_acc']:.2f}% at epoch {best_metrics['best_epoch']}")

        return best_metrics
