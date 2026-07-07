# Audit Log

Author: Richard Van Winkle, Matrikelnummer: 3214183

| ID | File | Problem / manifestation | Mathematical or logical root cause | Fix implemented | Commit hash |
|----|------|-------------------------|------------------------------------|-----------------|-------------|
| fix1 | data.py | Cannot read the data files, runtime error | There is a _data suffix after the file name ln. 11 | delete the suffix in code, so actual file names are brought in | 0263575 |
| fix2 | models.py | the first input to nn.crossEntropyLoss() was noneType, not Tensor | the forward() call in the ResNet18 class was not returning the images, automically assigning them to none | set self.classifer(out) to out, then returned out | 0255111 |