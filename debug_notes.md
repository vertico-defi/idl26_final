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