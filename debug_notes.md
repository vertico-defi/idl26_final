## fix1
```bash
(dlFinal_gpu) ┌──(vertico㉿vertico)-[~/thws/semester3/dlFinal]
└─$ python3 code/train.py
Training executing on device: cuda
Traceback (most recent call last):
  File "/home/vertico/thws/semester3/dlFinal/code/train.py", line 33, in <module>
    main()
  File "/home/vertico/thws/semester3/dlFinal/code/train.py", line 22, in main
    train_loader, val_loader, _ = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])
  File "/home/vertico/thws/semester3/dlFinal/code/data.py", line 12, in get_loaders
    data_dict = torch.load(d_path)
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/serialization.py", line 997, in load
    with _open_file_like(f, 'rb') as opened_file:
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/serialization.py", line 444, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/serialization.py", line 425, in __init__
    super().__init__(open(name, mode))
FileNotFoundError: [Errno 2] No such file or directory: 'data/lesions_data.pt'
```

### error notes

The code is adding a suffix to the data files, that is not neccessary, the files don't have that suffix. Let's change it in the code instead of altering the files.

## fix2


```bash
(dlFinal_gpu) ┌──(vertico㉿vertico)-[~/thws/semester3/dlFinal]
└─$ python3 code/train.py
Training executing on device: cuda
Using activation function: Identity()

 Starting Training Routine...
--------------------------------------------------
Traceback (most recent call last):
  File "/home/vertico/thws/semester3/dlFinal/code/train.py", line 33, in <module>
    main()
  File "/home/vertico/thws/semester3/dlFinal/code/train.py", line 30, in main
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
  File "/home/vertico/thws/semester3/dlFinal/code/fit.py", line 60, in fit
    train_loss, train_acc = self.train_one_epoch(train_loader)
  File "/home/vertico/thws/semester3/dlFinal/code/fit.py", line 24, in train_one_epoch
    loss = self.criterion(outputs, labels)
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/nn/modules/loss.py", line 1185, in forward
    return F.cross_entropy(input, target, weight=self.weight,
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/nn/functional.py", line 3086, in cross_entropy
    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not NoneType
                                                                                                                            
(dlFinal_gpu) ┌──(vertico㉿vertico)-[~/thws/semester3/dlFinal]

```

### error notes

The input to the crossEntropyLoss is the main type. I traced the crossEntropyLoss function back to the Trainer class through the criterion variable. Then, I should check the types of the inputs into critireon to see what those are, the inputs are `(outputs, labels), outputs is drawing it's value from the images

I added some print statements to see the type and values of outputs and I got this:

code:

```python
 for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            # (fix2) SANITY CHECK: This self.criterion is nn.CrossEntropyLoss() fromt train.py
            # (fix2) so these inputs can be viewed as inputs into the nn.CrossEntropyLoss() object
            # (fix2) and compares the predicted outputs to the labels. Check the outputs size here
            print("outputs type:", type(outputs)) # <- added
            print("outputs calue:", outputs) # <- added
            loss = self.criterion(outputs, labels)
```

Here is the output:

```bash
outputs type: <class 'NoneType'>
outputs calue: None
```


### Possible Magda questions for fix 2

What does the loss.backward and self.optimzer.step do from the fit.py file?

what does the self.model call do in the fit.py file.

Explain the forward pass from the REsNet18 class in the models.py file. 

What do I need to pass the output of the forward pass into the loss function?

What is a logit?

Explain the fdifference between a class, object, instance and method.

What do the __init__ thingies mean as definitions inside of classes?

## fix 3

Here is the error I've been getting now:

```bash
(dlFinal_gpu) ┌──(vertico㉿vertico)-[~/thws/semester3/dlFinal]
└─$ python3 code/train.py
Training executing on device: cuda
Using activation function: Identity()

 Starting Training Routine...
--------------------------------------------------
outputs type: <class 'torch.Tensor'>
outputs calue: tensor([[ 4.2683e-01, -2.6112e-01, -4.1677e-01,  8.3694e-01,  7.3020e-04,
         -5.4151e-01, -6.6031e-01,  1.0089e+00,  1.1035e+00,  2.9190e-01],
        [ 2.5610e-01, -3.5740e-01, -8.6032e-01, -1.3047e-01,  1.4557e-01,
         -2.1004e-01,  2.1943e-01,  6.9421e-01, -2.7911e-01, -4.1145e-01],
        [-1.1020e-01,  4.9030e-02,  2.8416e-01, -6.3529e-01, -2.2238e-01,
          2.2986e-01,  4.9303e-01, -6.4029e-01, -7.6662e-01, -2.6301e-01],
        [ 1.8509e-02, -1.9853e-01,  4.7008e-01, -2.4306e-01, -3.4761e-02,
          5.7778e-02,  2.9186e-01, -2.6624e-01, -8.8388e-02, -3.4910e-01],
        [-1.1854e-01,  2.6128e-01,  2.0186e-01,  3.6309e-01, -5.2636e-02,
         -8.0418e-02, -4.1235e-01, -1.6473e-03,  6.4643e-01,  4.3261e-01],
        [-1.6592e-01,  2.8623e-01,  5.2008e-01,  3.4780e-01, -1.5079e-01,
          3.1372e-02, -3.0145e-01, -4.1371e-01,  6.2376e-01,  3.1853e-01],
        [-3.7402e-01, -8.7180e-02,  8.7008e-01, -1.1323e+00, -1.5105e-01,
          7.3852e-01,  8.6729e-01, -1.0746e+00, -9.6880e-01, -5.1787e-01],
        [-1.5714e-01,  6.1078e-01, -1.3458e+00,  2.6763e-01,  1.9218e-01,
         -3.9470e-01, -2.2065e-01,  6.6544e-01, -1.5424e-01,  1.4788e-01]],
       device='cuda:0', grad_fn=<AddmmBackward0>)
Traceback (most recent call last):
  File "/home/vertico/thws/semester3/dlFinal/code/train.py", line 35, in <module>
    main()
  File "/home/vertico/thws/semester3/dlFinal/code/train.py", line 32, in main
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
  File "/home/vertico/thws/semester3/dlFinal/code/fit.py", line 69, in fit
    train_loss, train_acc = self.train_one_epoch(train_loader)
  File "/home/vertico/thws/semester3/dlFinal/code/fit.py", line 33, in train_one_epoch
    loss = self.criterion(outputs, labels)
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/nn/modules/loss.py", line 1185, in forward
    return F.cross_entropy(input, target, weight=self.weight,
  File "/home/vertico/miniconda3/envs/dlFinal_gpu/lib/python3.10/site-packages/torch/nn/functional.py", line 3086, in cross_entropy
    return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)
RuntimeError: 0D or 1D target tensor expected, multi-target not supported
                                                                                                                            
(dlFinal_gpu) ┌──(vertico㉿vertico)-[~/thws/semester3/dlFinal]

```

## Error notes

Now, it is saying that the target is supposed to be 0D or 1D, but we are getting multi-target. 

Luckily this is in the same ballpark as the issue we had before where the first input of the cross entropy loss was off, now it is the second parameter, so we can go straight to where we printed the first parameter and print the second one so that we can investigate further. 
```bash
(dlFinal_gpu) ┌──(vertico㉿vertico)-[~/thws/semester3/dlFinal]
└─$ python3 code/train.py
Training executing on device: cuda
Total number of samples: 8010
Number of training samples: 8010
Number of validation samples: 801
```

See above, while going to check the dimensions of the target parameter for the crossEntropyLoss function, I noticed that the validation data was properly getting split from the training data, but that the training data was not being split to reflect that. and that there would be overlap between the training and the validation data. Also, I found that the lavels source  in the data.py file and that we can use the squeeze object from pytorch to change the dimensions correctly.


### source on cross_entropy_loss function
https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html?utm_source=chatgpt.com

### source on the squeeze function
https://docs.pytorch.org/docs/2.12/generated/torch.squeeze.html?utm_source=chatgpt.com

### MAgda questions

What does the squeeze function do?

Why is it important that the training and validation data be split?

Why would you use -1 as a parameter for the squeeze function? and how can you be certain that the last value will always be one, because the squeeze function will only ever get rid of singular values.

## fix 4

```bash
Using activation function: Identity()

 Starting Training Routine...
--------------------------------------------------
Epoch [01/20] | Train Loss: 14.1793 - Train Acc: 45.58% | Val Loss: 37.0941 - Val Acc: 11.36%
Epoch [02/20] | Train Loss: 49.7563 - Train Acc: 48.58% | Val Loss: 58.1333 - Val Acc: 65.92%
Epoch [03/20] | Train Loss: 69.5729 - Train Acc: 48.08% | Val Loss: 148.4485 - Val Acc: 11.61%
Epoch [04/20] | Train Loss: 109.1330 - Train Acc: 49.45% | Val Loss: 95.4577 - Val Acc: 63.92%
Epoch [05/20] | Train Loss: 98.7724 - Train Acc: 46.82% | Val Loss: 64.2171 - Val Acc: 65.54%
Epoch [06/20] | Train Loss: 89.8022 - Train Acc: 49.11% | Val Loss: 43.2827 - Val Acc: 62.42%
Epoch [07/20] | Train Loss: 159.1353 - Train Acc: 44.85% | Val Loss: 260.7986 - Val Acc: 5.12%
Epoch [08/20] | Train Loss: 216.9113 - Train Acc: 48.73% | Val Loss: 202.2892 - Val Acc: 33.83%
Epoch [09/20] | Train Loss: 143.7413 - Train Acc: 48.73% | Val Loss: 93.3259 - Val Acc: 49.44%
Epoch [10/20] | Train Loss: 244.0868 - Train Acc: 52.06% | Val Loss: 332.1478 - Val Acc: 65.92%
Epoch [11/20] | Train Loss: 313.6982 - Train Acc: 48.95% | Val Loss: 423.7872 - Val Acc: 65.92%
Epoch [12/20] | Train Loss: 421.4338 - Train Acc: 50.13% | Val Loss: 355.1691 - Val Acc: 61.17%
Epoch [13/20] | Train Loss: 326.7316 - Train Acc: 51.01% | Val Loss: 359.2523 - Val Acc: 65.92%
Epoch [14/20] | Train Loss: 351.5818 - Train Acc: 47.00% | Val Loss: 340.4111 - Val Acc: 46.19%
Epoch [15/20] | Train Loss: 335.5813 - Train Acc: 52.66% | Val Loss: 196.5404 - Val Acc: 65.92%
Epoch [16/20] | Train Loss: 313.0668 - Train Acc: 44.68% | Val Loss: 269.1578 - Val Acc: 1.87%
Epoch [17/20] | Train Loss: 317.3027 - Train Acc: 48.26% | Val Loss: 273.1329 - Val Acc: 33.21%
Epoch [18/20] | Train Loss: 296.1408 - Train Acc: 48.34% | Val Loss: 285.8419 - Val Acc: 65.92%
Epoch [19/20] | Train Loss: 352.2904 - Train Acc: 45.58% | Val Loss: 302.4558 - Val Acc: 20.22%
Epoch [20/20] | Train Loss: 288.9888 - Train Acc: 50.89% | Val Loss: 483.1311 - Val Acc: 11.61%
--------------------------------------------------
```

## error notes

The training loss is relatively low on the first epoch then keeps growing, same with validation loss. The reason for this could be that the gradients are not getting zerod out between passes through the neural networks. The new weights at each point are calculated with these gradients and if you don't zero them they get accumulated, which means that at each new point the gradient may not point in the direction of the optimal weight because it is influenced by the gradient at each previous point. 

This actually turned out to work, because the training loss kept getting smaller:

```bash
(dlFinal_gpu) ┌──(vertico㉿vertico)-[~/thws/semester3/dlFinal]
└─$ python3 code/train.py                                                                                         
Training executing on device: cuda
Using activation function: Identity()

 Starting Training Routine...
--------------------------------------------------
Epoch [01/20] | Train Loss: 1.1525 - Train Acc: 63.73% | Val Loss: 1.0963 - Val Acc: 64.04%
Epoch [02/20] | Train Loss: 1.0346 - Train Acc: 65.36% | Val Loss: 1.0337 - Val Acc: 63.55%
Epoch [03/20] | Train Loss: 0.9963 - Train Acc: 66.00% | Val Loss: 0.9695 - Val Acc: 66.17%
Epoch [04/20] | Train Loss: 0.9745 - Train Acc: 66.42% | Val Loss: 0.9655 - Val Acc: 65.17%
Epoch [05/20] | Train Loss: 0.9582 - Train Acc: 67.07% | Val Loss: 0.9313 - Val Acc: 66.79%
Epoch [06/20] | Train Loss: 0.9494 - Train Acc: 66.99% | Val Loss: 0.9516 - Val Acc: 66.29%
Epoch [07/20] | Train Loss: 0.9447 - Train Acc: 66.79% | Val Loss: 0.9149 - Val Acc: 66.92%
Epoch [08/20] | Train Loss: 0.9329 - Train Acc: 67.44% | Val Loss: 0.9446 - Val Acc: 66.42%
Epoch [09/20] | Train Loss: 0.9314 - Train Acc: 67.08% | Val Loss: 0.9140 - Val Acc: 67.79%
Epoch [10/20] | Train Loss: 0.9255 - Train Acc: 67.79% | Val Loss: 0.8966 - Val Acc: 67.29%
Epoch [11/20] | Train Loss: 0.9184 - Train Acc: 67.26% | Val Loss: 0.9029 - Val Acc: 66.04%
Epoch [12/20] | Train Loss: 0.9128 - Train Acc: 67.92% | Val Loss: 0.9154 - Val Acc: 64.92%
Epoch [13/20] | Train Loss: 0.9070 - Train Acc: 68.22% | Val Loss: 0.9297 - Val Acc: 62.05%
Epoch [14/20] | Train Loss: 0.9054 - Train Acc: 67.55% | Val Loss: 0.8919 - Val Acc: 65.92%
Epoch [15/20] | Train Loss: 0.9062 - Train Acc: 67.97% | Val Loss: 0.8839 - Val Acc: 67.29%
Epoch [16/20] | Train Loss: 0.9027 - Train Acc: 68.00% | Val Loss: 0.9042 - Val Acc: 67.92%
Epoch [17/20] | Train Loss: 0.9022 - Train Acc: 68.30% | Val Loss: 0.9125 - Val Acc: 65.92%
Epoch [18/20] | Train Loss: 0.8980 - Train Acc: 67.85% | Val Loss: 0.8811 - Val Acc: 68.04%
Epoch [19/20] | Train Loss: 0.8917 - Train Acc: 68.26% | Val Loss: 0.8964 - Val Acc: 68.54%
Epoch [20/20] | Train Loss: 0.8988 - Train Acc: 68.12% | Val Loss: 0.8981 - Val Acc: 66.04%
--------------------------------------------------
Training Complete!
                                                                                                                            
(dlFinal_gpu) ┌──(vertico㉿vertico)-[~/thws/semester3/dlFinal]
└─$ 
```