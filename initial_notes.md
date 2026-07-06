# Intro Deep Learning Final Assignment

## Initial notes:

**Notes**

- Data has been downloaded and placed as a child directory underneath the root.
- The following have been **deleted** (by external party) 
    1. Trained weights, 
    2. production-ready source code, 
    3. configuration registries, 
    4. and evaluation automation scripts
- The following were **recovered**
    1. data.py
    2. models.py
    3. train.py
    4. trainer.py
- The files above have all been downloaded to the code/ directory inside of the root directory
- There is a Github repo, that has been copied over to my own github account in its own repo

### Deliverables

*Quick Overview*

- Audit the remains of the codebase
- Neutralize every mathematical and syntactic issue
- Create the missing configuration
- Test the infrastructure from scratch to restore project integrity
- Clear commit messages

1. **Complete Corrected Codebase**
    - Fully functional, production-grade version of entire repo.
        - data.py, models.py, train.py, and trainer.py
    - Also includes newly engineered configuration and testing frameworks.
    - Codebase must be modular, error-free, compatible with all four target datasets, and capable of executing training and predictions for all three models
        - via native Python calls driven by the developed configuration structure
2. **Professional Repository Documentation**
    - Comprehensive markdown file in the root of `main` branch detailing
        - content
        - structural layout
        - prerequisites
        - commands to install dependencies and execute the train and test pipeline

3. **Incident Audit Log**
    - Technical table submitted as `AUDIT_LOG.md`
        - itemize every bug, corruption, or anti-pattern discovered within the code
        - For each entry
            - file name
            - how the problem manifests
            - mathematical or logical root cause of the failure
            - structural correction implemented
            - exact git commit hash containing the fix

4. **Consolidated Benchmark Report**
    - A breif evaluation report submitted as `REPORT.md`
        - In repository root
    - Compare the final performance of the corrected models across all configuration permutations
        - Must include a clean summary table capturing key metrics
            - Accuracy, precision, recall, and macro F1-score
        - Also include architechtural recommendations for optimal dataset/model pairings based on your observed benchmarks.
    - Expected minimal accuracy:
        - Cells: 90%, chest: 87%, lesions: 67%, orgs: 83%
