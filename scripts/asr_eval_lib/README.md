# Architecture
Input:
a. configs (common, user specific)
b. HF datasets in BIGOS format

Output:
a. HF dataset with ASR hypotheses for set of ASR systems


Key features:
a. Prefect based orchestration
b. Parallel processing
c. Standardized interface for ASR systems enabling extendability
d. portable code/reusable code base
e. easy parametrization to support HF datasets in different format
f. easy backup of intermediary result in local and HF server based cache