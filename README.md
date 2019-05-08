# Semi-supervised VAE for Vulnerability Detection

Project for ECBM 6040 Neural Networks and Deep Learning Research

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Required packages are listed in [a text file](requirement.txt)

```
pip install -r requirements.txt
```

### Project Structure

The report is written as a [Jupyter Notebook](semi_vae.ipynb)

The script for cross generalization test that includes a CNN, LSTM, BiLSTM implementation

```
cross_trainer.py
cross_dataset.py
cross_model.py
```

The script for semi-supervised VAE

```
ss_vae.py
dataset_ss.py
vae_model.py
```

The data are in 

```
./data
```

The live demo is at http://35.231.70.60:9999

## Built With

* [PyTorch](https://pytorch.org/) - Deep Learning Framework
* [Pyro](https://pyro.ai/) - Probablistic Programming Framework


## Authors

* **Yufan Zhuang** - Data Science Institute, Columbia University

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Reference

[Kingma, Durk P., et al. "Semi-supervised learning with deep generative models." Advances in neural information processing systems. 2014.](https://arxiv.org/abs/1406.5298)
