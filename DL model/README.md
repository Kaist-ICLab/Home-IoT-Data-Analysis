## Getting Started
Install the following libraries
- pip install cloudpickle
- pip install transformers, datasets
- pip install accelerate -U
- pip install shap
- Python==3.9
- NumPy
- pandas==2.1.1
- scikit-learn
- pytorch

## Running the Code
```python
  python main.py --modalname conv1dattn # single experiment
  python run_models.py # multi experiments
```
#### modelname
```python
  models = {
      'conv1d': CNN1d
  }
```

#### model training parameters
```python
#TODOs
```