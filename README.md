# Lorentz Boost Network (LBN) [![pipeline status](https://git.rwth-aachen.de/3pia/lbn/badges/master/pipeline.svg)](https://git.rwth-aachen.de/3pia/lbn/pipelines)

TensorFlow implementation of the Lorentz Boost Network from [arXiv:1812.09722 [hep-ex]](https://arxiv.org/abs/1812.09722).

Original repository: [git.rwth-aachen.de/3pia/lbn](https://git.rwth-aachen.de/3pia/lbn)


### Usage example

```python
from lbn import LBN

# initialize the LBN, set 10 combinations and pairwise boosting
lbn = LBN(10, boost_mode=LBN.PAIRS)

# create a feature tensor based on input four-vectors
features = lbn(four_vectors)

# use the features as input for a subsequent, application-specific network
...
```


### Installation and dependencies

Via [pip](https://pypi.python.org/pypi/lbn):

```bash
pip install lbn
```

NumPy and TensorFlow are the only dependencies.


### Testing

Tests should be run for Python 2 and 3. The following commands assume you are root directory of the LBN respository:

```bash
python -m unittest test

# or via docker, python 2
docker run --rm -v `pwd`:/root/lbn -w /root/lbn tensorflow/tensorflow:latest python -m unittest test

# or via docker, python 3
docker run --rm -v `pwd`:/root/lbn -w /root/lbn tensorflow/tensorflow:latest-py3 python -m unittest test
```


### Contributing

If you like to contribute, we are happy to receive pull requests. Just make sure to add new test cases and run the tests. Also, please use a coding style that is compatible with our `.flake8` config.


### Development

- Original source hosted at [RWTH GitLab](https://git.rwth-aachen.de/3pia/lbn)
- Report issues, questions, feature requests on [RWTH GitLab Issues](https://git.rwth-aachen.de/3pia/lbn/issues)
