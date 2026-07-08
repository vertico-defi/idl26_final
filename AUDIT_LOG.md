# Audit Log

Author: Richard Van Winkle, Matrikelnummer: 3214183

| ID | File | Problem / manifestation | Mathematical or logical root cause | Fix implemented | Commit hash |
|----|------|-------------------------|------------------------------------|-----------------|-------------|
| fix1 | data.py | Cannot read the data files, runtime error | There is a _data suffix after the file name ln. 11 | delete the suffix in code, so actual file names are brought in | 0263575 |
| fix2 | models.py | the first input to nn.crossEntropyLoss() was noneType, not Tensor | the forward() call in the ResNet18 class was not returning the images, automically assigning them to none | set self.classifer(out) to out, then returned out | 0255111 |
| fix3 | data.py | the target parameter to nn.crossEntropyLoss was the wrong shape | If first input is shape (N,C) the target input must be of shape (N), it was (N, 1) | used torch.squeeze(-1) object to shape the target correctly | ba4fc39 |
| fix3 | data.py | training and validation splits overlapped | training smaple size was the same as total sample size, not a 90/10 split | seperated training data using the val_split variable | ba4fc39 |
| fix4 | fit.py | train and val loss exploded in value as the epochs increased | The gradients were not getting zeroed after each pass through the NN, causing the weights to reuse/accumulate gradients and wander aimlessly, i.e. to not optimize | I added the optimizer.zero_grad() function right before the self.model call | 62b2833 |
| fix5 | models.py | ResNet18 training accuracy was around 66 - 68% | ResNEt18 model was using Identity() as the activation function, so there was no non-linear activation. The ResNet paper says identity refers to the shortcut connection and that ReLU is used for activation | changed activation_str to ReLU, validation accuracy improved to 75.3% | 78c1d44 |