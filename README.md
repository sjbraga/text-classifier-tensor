# Classificador de texto com Deep Learning


## Dependências

- Python 2.7
- TensorFlow 1.4.0
- Numpy
## Uso

    python train.py

O treinamento com todos os exemplos no dataset leva em torno de 1h30, para executar o treinamento com conjunto menor de treinamento executar como:

    python train.py --max_dataset_inputs=1000

Para utilizar regularização L2 utilizar:

    python train.py --l2_reg_lambda=0.1