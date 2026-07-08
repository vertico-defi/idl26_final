"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    d_path = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path)

    total_samples = data_dict['train_images'].shape[0]
    """ (fix3) while we are here and suspect a discrepency in the train and validation sizes, let's
    print the number of total samples and see that that is the sum of the amount of training and validation
    samples.    
    """
    # print("Total number of samples:", total_samples)
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    """ (fix3) So here are the labels, let's confirm the shape here is off. Also, it looks
    like the training data is getting split into training and validation data, but there is
    no split occuring for the training data itself. According the documentation of the 
    nn.crossEntropyLoss function (see debug notes for the article snippet), if the input, or first parameter of the object
    is of size (N,C) where N is the batch_size and C is the number of classes, then the target should be of size (N), but 
    the error was saying it was of size (8,1), which is incorrect. We'll use the squeeze(-1) function to shape the data correctly
    """
    train_data = data_dict['train_images'][:val_start]
    # (fix3) check the train_data size
    # print("Number of training samples:", train_data.shape[0])

    train_labels = data_dict['train_labels'][:val_start].squeeze(-1)
    
    
    val_data = data_dict['train_images'][val_start:]
    # (fix3) check the size of the validation data too
    # print("Number of validation samples:", val_data.shape[0])
    val_labels = data_dict['train_labels'][val_start:].squeeze(-1)
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(data_dict['test_images'], data_dict['test_labels'])
    
    # (fix3) here is train_loader, it looks like it is a collection of the training data and labels
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader