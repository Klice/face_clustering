# Face clustering

Python 3.9+

## Install

```
pipenv install
```

## Running

### Step 1 - Analyze images

Put images into `images/` folder (see `IMAGES_PATH` in `config.py`) and run `analyze.py`. It will run face detection and face recognitions for all images in `IMAGES_PATH` folder. The results will be stored in `DATA_FILE` file.
If new images added, re-run `analyze.py`, it will perform analysis only for added images.

### Step 2 - Use Jupyter notebook for cluster analysis

`cluster.ipynb` contains the following steps:
1. Read data from `DATA_FILE` file
2. Perform analysis to determin optimal number of cluster
3. Group images into specified (`n_clusters` variable) number of clusters