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