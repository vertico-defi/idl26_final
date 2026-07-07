# Audit Log

Author: Richard Van Winkle, Matrikelnummer: 3214183

| ID | File | Problem / manifestation | Mathematical or logical root cause | Fix implemented | Commit hash |
|----|------|-------------------------|------------------------------------|-----------------|-------------|
| 1 | data.py | Cannot read the data files, runtime error | There is a _data suffix after the file name ln. 11 | delete the suffix in code, so actual file names are brought in | 0263575 |