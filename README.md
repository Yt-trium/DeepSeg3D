# DeepSeg3D
Neural network toolkit for 3D image segmentation

#### All the necessary information and documentation is available on the [WIKI](https://github.com/Yt-trium/DeepSeg3D/wiki) !

```
+---------------+
|               +-------+ Input images
|     Input     |
|               |
+-------+-------+
        |
+-------v-------+
|               +-------+ Compute parameters with train images
| Preprocessing |
|               |
+-------+-------+
        |
+-------v-------+
|               <-------+
|     Train     |       | Training with ground truth
|               +-------+
+-------+-------+
        |
+-------v-------+
|               +-------+ Compute parameters with train images
|Postprocessing |
|               |
+---------------+
```

# Training
To run the training, launch [training.py](training.py) with python3.

* Limit GPU usage : CUDA_VISIBLE_DEVICES=0,1,2,...

* Use CPU only : CUDA_VISIBLE_DEVICES=''

* Example of training
```
CUDA_VISIBLE_DEVICES=0,1 nohup python3 training.py ./config.txt &
```


# Testing
To run the testing, launch [testing.py](testing.py) with python3.

* Example of test
```
CUDA_VISIBLE_DEVICES=2,3 python3 testing.py ./config.txt ./logs/model-001.h5
```


# Examples
* Unet
* Dolz


# Miscellaneous

* remove 90% of saved models in logs folders
```
rm ./logs*/model-[0-9][0-9][1-9]*
```


# Future work

```
                 +---------------+
                 |               +-------+ Input images
                 |     Input     |
                 |               |
                 +-------+-------+
+---------------+        |        +---------------+
|               |        |        |               |
|  Full images  +<-------+------->+    Patchs     |
|               |                 |               |
+-------+-------+                 +-------+-------+
        |                                 |
+-------v-------+                 +-------v-------+
|               |                 |               |
|     Train     |                 |     Train     |
|               |                 |               |
+-------+-------+                 +-------+-------+
        |                                 |
+-------v-------+                 +-------v-------+
|               |                 |               |
|  Prediction   |                 |  Prediction   |
|               |                 |               |
+-------+-------+                 +-------+-------+
        |        +---------------+        |
        |        |               |        |
        +-------->     Merge     <--------+
                 |               |
                 +-------+-------+
                         |
                 +-------v-------+
                 |               |
                 |     Train     |
                 |               |
                 +-------+-------+
                         |
                 +-------v-------+
                 |               |
                 |     Output    |
                 |               |
                 +---------------+
```