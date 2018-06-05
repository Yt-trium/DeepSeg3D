# ------------------------------------------------------------ #
#
# file : preprocessing/normalisation.py
# author : CM
# Preprocess dataset with intensity normalisation
# (zero mean and unit variance)
# ------------------------------------------------------------ #

def intensityNormalisation(dataset, dtype):
    mean = dataset.mean()
    std  = dataset.std()
    return ((dataset - mean) / std).astype(dtype)