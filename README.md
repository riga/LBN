# Lorentz Boost Network (LBN) [![pipeline status](https://git.rwth-aachen.de/3pia/lbn/badges/master/pipeline.svg)](https://git.rwth-aachen.de/3pia/lbn/pipelines)

TensorFlow implementation of the Lorentz Boost Network from [arXiv:1812.09722 [hep-ex]](https://arxiv.org/abs/1812.09722).

Original repository: [git.rwth-aachen.de/3pia/lbn](https://git.rwth-aachen.de/3pia/lbn)


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
model.add(LBNLayer(10, boost_mode=LBN.PAIRS))

# add a dense layer
model.add(tf.keras.layers.Dense(1024))

# continue builing and training the model
...
```


### Installation and dependencies

Via [pip](https://pypi.python.org/pypi/lbn):

```bash
pip install lbn
```

NumPy and TensorFlow are the only dependencies. Both TensorFlow v1 and v2 are supported.


### Testing

Tests should be run for Python 2 and 3 and for TensorFlow 1 and 2. The following commands assume you are root directory of the LBN respository:

```bash
python -m unittest test

# or via docker, python 2 and tf 1
docker run --rm -v `pwd`:/root/lbn -w /root/lbn tensorflow/tensorflow:1.13.1 python -m unittest test

# or via docker, python 3 and tf 2
docker run --rm -v `pwd`:/root/lbn -w /root/lbn tensorflow/tensorflow:2.0.0a0-py3 python -m unittest test
```


### Contributing

If you like to contribute, we are happy to receive pull requests. Just make sure to add new test cases and run the tests. Also, please use a coding style that is compatible with our `.flake8` config.


### Development

- Original source hosted at [RWTH GitLab](https://git.rwth-aachen.de/3pia/lbn)
- Report issues, questions, feature requests on [RWTH GitLab Issues](https://git.rwth-aachen.de/3pia/lbn/issues)
