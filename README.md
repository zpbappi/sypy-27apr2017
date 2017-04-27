# sypy-27apr2017
Presentation and demo materials presented on 27th April 2017 in Sydney Python User Group (SyPy)

## Getting CNTK
Please go to the [CNTK documentation](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine) for instruction on how to setup in your machine.

## Running the example

This is a two part operation. Getting the data first and then running the training.

### Getting the MNIST data
You only need to do this once. MNIST data is downloaded, converted into CNTK Text Format and saved for later use. Go into the `/datasets/MNIST` directory and run 
```python
python install_mnist.py
```

### Training the model
From the root directory, run
```python
python mnist.py
```

Note: This code is tested in python 3.4. If you are using python 2.x, please add some _futuristic imports_ wherever necessary.

## Further reading
Probably the richest resource right now is the [wiki](https://github.com/Microsoft/CNTK/wiki).  