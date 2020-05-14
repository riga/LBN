# Lorentz Boost Network (LBN) [![Build Status](https://travis-ci.org/riga/LBN.svg?branch=master)](https://travis-ci.org/riga/LBN) [![Package Status](https://badge.fury.io/py/lbn.svg)](https://badge.fury.io/py/lbn)

TensorFlow implementation of the Lorentz Boost Network from [arXiv:1812.09722 [hep-ex]](https://arxiv.org/abs/1812.09722).


### Usage example

```python
import tensorflow as tf
from lbn import LBN

# initialize the LBN, set 10 combinations and pairwise boosting
lbn = LBN(10, boost_mode=LBN.PAIRS)

# create a feature tensor based on input four-vectors
features = lbn(four_vectors)

# use the features as input for a subsequent, application-specific network
...
```

Or with TensorFlow 2 and Keras:

```python
import tensorflow as tf
from lbn import LBN, LBNLayer

# start a sequential model
model = tf.keras.models.Sequential()

# add the LBN layer
input_shape = (6, 4)
model.add(LBNLayer(input_shape, 10, boost_mode=LBN.PAIRS))

# add a dense layer
model.add(tf.keras.layers.Dense(1024))

# continue builing and training the model
...
```

For more examples on how to set up the LBN with TensorFlow (eager mode and autograph / `tf.function` ) and Keras, see [this gist](https://gist.github.com/riga/fe13cc42605547adcecb9b92484f06db).


### Installation and dependencies

Via [pip](https://pypi.python.org/pypi/lbn):

```bash
pip install lbn
```

NumPy and TensorFlow are the only dependencies. Both TensorFlow v1 and v2 are supported.


### Testing

Tests should be run for Python 2 and 3 and for TensorFlow 1 and 2. The following commands assume you are in the root directory of the LBN respository:

```bash
python -m unittest test

# or via docker
for tag in 1.15.2 1.15.2-py3 2.1.0 2.2.0; do
    docker run --rm -v `pwd`:/root/lbn -w /root/lbn tensorflow/tensorflow:$tag python -m unittest test
done
```


### Contributing

If you like to contribute, we are happy to receive pull requests. Just make sure to add new test cases and run the tests. Also, please use a coding style that is compatible with our `.flake8` config.


### Development

- Original source hosted on [GitHub](https://github.com/riga/LBN)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/riga/LBN/issues)
