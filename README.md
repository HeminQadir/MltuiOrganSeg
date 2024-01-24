# MltuiOrganSeg


## Checking the libraries 
- To check the libraries and thier versions you can run 
```shell
python display_config.py
```


## Train model 

- To lunch training the model run the following 

```shell
python train.py
```


- To lunch tensorboard run 

```shell
tensorboard --logdir="path/to/log/"
```

You can open `train.py` and change the hyper-parameters and others things such as the save_dir and data_dir. Later, I will add args for these hyper-parameters and paths 

## Note:
- This represents only the initial version, with numerous modifications in progress
