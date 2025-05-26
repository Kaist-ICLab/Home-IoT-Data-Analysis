## Getting Started
Install the following libraries
- pip install cloudpickle
- pip install transformers, datasets
- pip install accelerate -U
- Python==3.9
- NumPy
- pandas==2.1.1
- scikit-learn
- pytorch
- pytorch-tabnet
- tab_transformer_pytorch
- hyperopt

## Running the Code
```python
  python main.py --modalname conv1dattn # single experiment
  python run_models.py # multi experiments
```
#### modelname
```python
  models = {
      'dnn_s': DNN_small,
      'dnn_m': DNN_medium,
      # 'convlstm': ConvLSTM,
      'conv1dcat': CNN1dCat,
      'conv1d': CNN1d,
      'conv1dattn': CNN1dAttn,
      # 'resnet': ResNet50,
  }
```

#### model training parameters
```python
#TODOs
```