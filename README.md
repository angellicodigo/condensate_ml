# condensate_ml

```bash
ml cudatoolkit/12.6                # If working on a cluster
export CONDA_OVERRIDE_CUDA="12.6"  # If working on a cluster
conda env create -f environment.yml
conda activate condenstate

pip install -e .
```

If there are changes to the environment.yml, you need to update your Anaconda environment
```bash
conda env update -f environment.yml --prune
```

