# coding: utf-8

"""
TensorFlow implementation of the Lorentz Boost Network (LBN). https://arxiv.org/abs/1812.09722.
"""


__author__ = "Marcel Rieger"
__copyright__ = "Copyright 2018-2020, Marcel Rieger"
__license__ = "BSD"
__credits__ = ["Martin Erdmann", "Erik Geiser", "Yannik Rath", "Marcel Rieger"]
__contact__ = "https://github.com/riga/LBN"
__email__ = "marcel.rieger@cern.ch"
__version__ = "1.2.1"

__all__ = ["LBN", "LBNLayer", "FeatureFactoryBase", "FeatureFactory"]


import functools

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops


# tf version flag
TF2 = tf.__version__.startswith("2.")


class LBN(object):
    """
    Lorentz Boost Network (LBN) class.

    Usage example:

    .. code-block:: python

        # initialize the LBN
        lbn = LBN(10, boost_mode=LBN.PAIRS)

        # create a feature tensor based on input four-vectors
        features = lbn(four_vectors)

        # use the features as input for a subsequent, application-specific network
        ...

    *n_particles* and *n_restframes* are the number of particle and rest-frame combinations to
    build. Their interpretation depends on the *boost_mode*. *n_restframes* is only used for the
    *PRODUCT* mode. It is inferred from *n_particles* for *PAIRS* and *COMBINATIONS*.

    *epsilon* is supposed to be a small number that is used in various places for numerical
    stability. When not *None*, *seed* is used to seed random number generation for trainable
    weights. *trainable* is passed to *tf.Variable* during weight generation. *name* is the main
    namespace of the LBN and defaults to the class name.

    *feature_factory* must be a subclass of :py:class:`FeatureFactoryBase` and provides the
    available, generic mappings from boosted particles to output features of the LBN. If *None*, the
    default :py:class:`FeatureFactory` is used.

    *particle_weights* and *restframe_weights* can refer to externally defined variables with custom
    initialized weights. If set, their shape must match the number of combinations and inputs. For
    simple initialization tests, *weight_init* can be a tuple containing the Gaussian mean and
    standard deviation that is passed to ``tf.random.normal``. When *None*, and the weight tensors
    are created internally, mean and standard deviation default to *0* and *1 / combinations*. When
    *abs_particle_weights* (*abs_restframe_weights*) is *True*, ``tf.abs`` is applied to the
    particle (rest frame) weights. When *clip_particle_weights* (*clip_restframe_weights*) is
    *True*, particle (rest frame) weights are clipped at *epsilon*, or at the passed value if it is
    not a boolean. Note that the abs operation is applied before clipping.

    When the number of features per input particle is larger than four, the subsequent values are
    interpreted as auxiliary features. Similar to the combined particles and restframes, these
    features are subject to linear combinations to create new, embedded representations. The number
    number of combinations, *n_auxiliaries*, defaults to the number of boosted output particles.
    Their features are concatenated to the vector of output features. The weight tensor
    *aux_weights* is used to create the combined feautres. When given, it should have the shape
    ``(n_in * (n_dim - 4)) x n_auxiliaries``.

    Instances of this class store most of the intermediate tensors (such as inputs, combinations
    weights, boosted particles, boost matrices, raw features, etc) for later inspection. Note that
    most of these tensors are set after :py:meth:`build` (or the :py:meth:`__call__` shorthand as
    shown above) are invoked.
    """

    # available boost modes
    PAIRS = "pairs"
    PRODUCT = "product"
    COMBINATIONS = "combinations"

    def __init__(self, n_particles, n_restframes=None, n_auxiliaries=None, boost_mode=PAIRS,
            feature_factory=None, particle_weights=None, abs_particle_weights=True,
            clip_particle_weights=False, restframe_weights=None, abs_restframe_weights=True,
            clip_restframe_weights=False, aux_weights=None, weight_init=None, epsilon=1e-5,
            seed=None, trainable=True, name=None):
        super(LBN, self).__init__()

        # determine the number of output particles, which depends on the boost mode
        # PAIRS:
        #   n_restframes set to n_particles, boost pairwise, n_out = n_particles
        # PRODUCT:
        #   boost n_particles into n_restframes, n_out = n_partiles * n_restframes
        # COMBINATIONS:
        #   build only particles, boost them into each other, except for boosts of particles into
        #   themselves, n_out = n**2 - n
        if boost_mode == self.PAIRS:
            n_restframes = n_particles
            self.n_out = n_particles
        elif boost_mode == self.PRODUCT:
            self.n_out = n_particles * n_restframes
        elif boost_mode == self.COMBINATIONS:
            n_restframes = n_particles
            self.n_out = n_particles**2 - n_particles
        else:
            raise ValueError("unknown boost_mode '{}'".format(boost_mode))

        # store boost mode and number of particles and restframes to build
        self.boost_mode = boost_mode
        self.n_particles = n_particles
        self.n_restframes = n_restframes
        self.n_auxiliaries = n_auxiliaries or self.n_out

        # particle weights and settings
        self.particle_weights = particle_weights
        self.abs_particle_weights = abs_particle_weights
        self.clip_particle_weights = clip_particle_weights
        self.final_particle_weights = None

        # rest frame weigths and settings
        self.restframe_weights = restframe_weights
        self.abs_restframe_weights = abs_restframe_weights
        self.clip_restframe_weights = clip_restframe_weights
        self.final_restframe_weights = None

        # auxiliary weights
        self.aux_weights = aux_weights

        # custom weight init parameters in a tuple (mean, stddev)
        self.weight_init = weight_init

        # epsilon for numerical stability
        self.epsilon = epsilon

        # random seed
        self.seed = seed

        # trainable flag
        self.trainable = trainable

        # internal name
        self.name = name or self.__class__.__name__

        # sizes that are set during build
        self.n_in = None  # number of input particles
        self.n_dim = None  # size per input vector, must be four or higher
        self.n_aux = None  # size of auxiliary features per input vector (n_dim - 4)

        # constants
        self.I = None  # the I matrix
        self.U = None  # the U matrix

        # tensor of input vectors
        self.inputs = None

        # split input tensors
        self.inputs_E = None  # energy column of inputs
        self.inputs_px = None  # px column of inputs
        self.inputs_py = None  # py column of inputs
        self.inputs_pz = None  # pz column of inputs
        self.inputs_aux = None  # auxiliary columns of inputs

        # tensors of particle combinations
        self.particles_E = None  # energy column of combined particles
        self.particles_px = None  # px column of combined particles
        self.particles_py = None  # py column of combined particles
        self.particles_pz = None  # pz column of combined particles
        self.particles_pvec = None  # p vectors of combined particles
        self.particles = None  # stacked 4-vectors of combined particles

        # tensors of rest frame combinations
        self.restframes_E = None  # energy column of combined restframes
        self.restframes_px = None  # px column of combined restframes
        self.restframes_py = None  # py column of combined restframes
        self.restframes_pz = None  # pz column of combined restframes
        self.restframes_pvec = None  # p vectors of combined restframes
        self.restframes = None  # stacked 4-vectors of combined restframes

        # Lorentz boost matrix (batch, n_out, 4, 4)
        self.Lambda = None

        # boosted particles (batch, n_out, 4)
        self.boosted_particles = None

        # features
        self.n_features = None  # total number of produced features
        self.boosted_features = None  # features of boosted particles
        self.aux_features = None  # auxiliary features (batch, n_in * n_aux, n_auxiliaries)
        self.features = None  # final, combined output features

        # initialize the feature factory
        if feature_factory is None:
            feature_factory = FeatureFactory
        elif not issubclass(feature_factory, FeatureFactoryBase):
            raise TypeError("feature_factory '{}' is not a subclass of FeatureFactoryBase".format(
                feature_factory))
        self.feature_factory = feature_factory(self)

        # the function that either builds the graph lazily, or can be used as an eager callable
        self._op = None

    @property
    def built(self):
        return self._op is not None

    @property
    def available_features(self):
        """
        Shorthand to access the list of available features in the :py:attr:`feature_factory`.
        """
        return list(self.feature_factory._feature_funcs.keys())

    def __call__(self, inputs, **kwargs):
        """
        Returns the LBN output features for specific *inputs*. It is ensured that the graph or eager
        callable are lazily created the first time this method is called by forwarding both *inputs*
        and *kwargs* to :py:meth:`build`.
        """
        # make sure the lbn op is built
        if not self.built:
            self.build(inputs.shape, **kwargs)

        # invoke it
        return self._op(inputs)

    def build(self, input_shape, features=("E", "px", "py", "pz"), external_features=None):
        """
        Builds the LBN structure layer by layer within dedicated variable scopes. *input_shape* must
        be a list, tuple or TensorShape object describing the dimensions of the input four-vectors.
        *features* and *external_features* are forwarded to :py:meth:`build_features`.
        """
        with tf.name_scope(self.name):
            # store shape and size information
            self.infer_sizes(input_shape)

            # setup variables
            with tf.name_scope("variables"):
                self.setup_weight("particle", (self.n_in, self.n_particles), 1)

                if self.boost_mode != self.COMBINATIONS:
                    self.setup_weight("restframe", (self.n_in, self.n_restframes), 2)

                if self.n_aux > 0:
                    self.setup_weight("aux", (self.n_in, self.n_auxiliaries, self.n_aux), 3)

            # constants
            with tf.name_scope("constants"):
                self.build_constants()

        # compute the number of total features
        self.n_features = 0
        # lbn features
        for feature in features:
            self.n_features += self.feature_factory._feature_funcs[feature]._shape_func(self.n_out)
        # auxiliary features
        if self.n_aux > 0:
            self.n_features += self.n_out * self.n_aux
        # external features
        if external_features is not None:
            self.n_features += external_features.shape[1]

        # also store the op that can be used to either create a graph or an eager callable
        def op(inputs):
            with tf.name_scope(self.name):
                with tf.name_scope("inputs"):
                    self.handle_input(inputs)

                with tf.name_scope("particles"):
                    self.build_combinations("particle")

                # rest frames are not built for COMBINATIONS boost mode
                if self.boost_mode != self.COMBINATIONS:
                    with tf.name_scope("restframes"):
                        self.build_combinations("restframe")

                with tf.name_scope("boost"):
                    self.build_boost()

                with tf.name_scope("features"):
                    if self.n_aux > 0:
                        with tf.name_scope("auxiliary"):
                            self.build_auxiliary()

                    self.build_features(features=features, external_features=external_features)

            return self.features

        self._op = op

    def infer_sizes(self, input_shape):
        """
        Infers sizes based on the shape of the input tensor.
        """
        if not isinstance(input_shape, (tuple, list, tf.TensorShape)):
            input_shape = input_shape.shape

        self.n_in = int(input_shape[-2])
        self.n_dim = int(input_shape[-1])

        if self.n_dim < 4:
            raise Exception("input dimension must be at least 4")
        self.n_aux = self.n_dim - 4

    def setup_weight(self, prefix, shape, seed_offset=0):
        """
        Sets up the variable tensors representing linear coefficients for the combinations of
        particles and rest frames. *prefix* must either be ``"particle"``, ``"restframe"`` or
        ``"aux"``. *shape* describes the shape of the weight variable to create. When not *None*,
        the seed attribute of this instance is incremented by *seed_offset* and passed to the
        variable constructor.
        """
        if prefix not in ["particle", "restframe", "aux"]:
            raise ValueError("unknown prefix '{}'".format(prefix))

        # define the weight name
        name = "{}_weights".format(prefix)

        # when the variable is already set, i.e. passed externally, validate the shape
        # otherwise, create a new variable
        W = getattr(self, name, None)
        if W is not None:
            # verify the shape
            w_shape = tuple(W.shape.as_list())
            if w_shape != shape:
                raise ValueError("the shape of variable {} {} does not match {}".format(
                    name, shape, w_shape))
        else:
            # define mean and stddev of weight init
            if isinstance(self.weight_init, tuple):
                mean, stddev = self.weight_init
            else:
                mean, stddev = 0., 1. / shape[1]

            # apply the seed offset when not None
            seed = (self.seed + seed_offset) if self.seed is not None else None

            # create and save the variable
            W = tf.Variable(tf.random.normal(shape, mean, stddev, dtype=tf.float32,
                seed=seed), name=name, trainable=self.trainable)
            setattr(self, name, W)

    def build_constants(self):
        """
        Builds the internal constants for the boost matrix.
        """
        # 4x4 identity
        self.I = tf.constant(np.identity(4), tf.float32)

        # U matrix
        self.U = tf.constant([[-1, 0, 0, 0]] + 3 * [[0, -1, -1, -1]], tf.float32)

    def handle_input(self, inputs):
        """
        Takes the passed *inputs* and stores internal tensors for further processing and later
        inspection.
        """
        # store the input vectors
        self.inputs = inputs

        # also store the four-vector components
        self.inputs_E = self.inputs[..., 0]
        self.inputs_px = self.inputs[..., 1]
        self.inputs_py = self.inputs[..., 2]
        self.inputs_pz = self.inputs[..., 3]

        # split auxiliary inputs
        if self.n_aux > 0:
            self.inputs_aux = self.inputs[..., 4:]

    def build_combinations(self, prefix):
        """
        Builds the combination layers which are quite similiar for particles and rest frames. Hence,
        *prefix* must be either ``"particle"`` or ``"restframe"``.
        """
        if prefix not in ("particle", "restframe"):
            raise ValueError("unknown prefix '{}'".format(prefix))

        # name helper
        name = lambda tmpl: tmpl.format(prefix)

        # get the weight tensor
        W = getattr(self, name("{}_weights"))

        # apply abs
        if getattr(self, name("abs_{}_weights")):
            W = tf.abs(W, name=name("abs_{}_weights"))

        # apply clipping
        clip = getattr(self, name("clip_{}_weights"))
        if clip is True:
            clip = self.epsilon
        if clip is not False:
            W = tf.maximum(W, clip, name=name("clipped_{}_weights"))

        # assign a name to the final weights
        W = tf.identity(W, name=name("final_{}_weights"))

        # create four-vectors of combinations
        E = tf.matmul(self.inputs_E, W, name=name("{}s_E"))
        px = tf.matmul(self.inputs_px, W, name=name("{}s_px"))
        py = tf.matmul(self.inputs_py, W, name=name("{}s_py"))
        pz = tf.matmul(self.inputs_pz, W, name=name("{}s_pz"))

        # create the full 3- and 4-vector stacks again
        p = tf.stack([px, py, pz], axis=-1, name=name("{}s_pvec"))
        q = tf.stack([E, px, py, pz], axis=-1, name=name("{}s"))

        # save all tensors for later inspection
        setattr(self, name("final_{}_weights"), W)
        setattr(self, name("{}s_E"), E)
        setattr(self, name("{}s_px"), px)
        setattr(self, name("{}s_py"), py)
        setattr(self, name("{}s_pz"), pz)
        setattr(self, name("{}s_pvec"), p)
        setattr(self, name("{}s"), q)

    def build_boost(self):
        """
        Builds the boosted particles depending on the requested boost mode. For infos on the boost
        matrix, see `this link <https://en.wikipedia.org/wiki/Lorentz_transformation>`__. The
        vectorized implementation is as follows:

        I = identity(4x4)

        U = -1(1x1)  0(1x3)
             0(3x1) -1(3x3)

        e = (1, -beta_vec/beta(1x3))^T

        Lambda = I + (U + gamma) x ((U + 1) x beta - U) x e . e^T
        """
        # n_particles and n_restframes must be identical for PAIRS and COMBINATIONS boosting
        if self.boost_mode in (self.PAIRS, self.COMBINATIONS):
            if self.n_restframes != self.n_particles:
                raise ValueError("n_restframes ({}) must be identical to n_particles ({}) in boost"
                    " mode '{}'".format(self.n_restframes, self.n_particles, self.boost_mode))

        # get the objects that are used to infer beta and gamma for the build the boost matrix,
        if self.boost_mode == self.COMBINATIONS:
            restframes_E = self.particles_E
            restframes_pvec = self.particles_pvec
        else:
            restframes_E = self.restframes_E
            restframes_pvec = self.restframes_pvec

        # to build the boost parameters, reshape E and p tensors so that batch and particle axes
        # are merged, and once the Lambda matrix is built, this reshape is reverted again
        # note: there might be more performant operations in future TF releases
        E = tf.reshape(restframes_E, [-1, 1])
        pvec = tf.reshape(restframes_pvec, [-1, 3])

        # for the boost to work, E must always be larger than p
        p = tf.reduce_sum(pvec**2., axis=1, keepdims=True)**0.5
        E = tf.maximum(E, p + self.epsilon)

        # determine the beta vectors
        betavec = pvec / E

        # determine the scalar beta and gamma values
        beta = tf.sqrt(tf.reduce_sum(tf.square(pvec), axis=1)) / tf.squeeze(E, axis=-1)
        gamma = 1. / tf.sqrt(1. - tf.square(beta) + self.epsilon)

        # the e vector, (1, -betavec / beta)^T
        beta = tf.expand_dims(beta, axis=-1)
        e = tf.expand_dims(tf.concat([tf.ones_like(E), -betavec / beta], axis=-1), axis=-1)
        e_T = tf.transpose(e, perm=[0, 2, 1])

        # finally, the boost matrix
        beta = tf.expand_dims(beta, axis=-1)
        gamma = tf.reshape(gamma, [-1, 1, 1])
        Lambda = self.I + (self.U + gamma) * ((self.U + 1) * beta - self.U) * tf.matmul(e, e_T)

        # revert the merging of batch and particle axes
        Lambda = tf.reshape(Lambda, [-1, self.n_restframes, 4, 4])

        # prepare particles for matmul
        particles = tf.reshape(self.particles, [-1, self.n_particles, 4, 1])

        # Lambda and particles need to be updated for PRODUCT and COMBINATIONS boosting
        if self.boost_mode in (self.PRODUCT, self.COMBINATIONS):
            # two approaches are possible
            # a) tile Lambda while repeating particles
            # b) batched gather using tiled and repeated indices
            # go with b) for the moment since diagonal entries can be removed before the matmul
            l_indices = np.tile(np.arange(self.n_restframes), self.n_particles)
            p_indices = np.repeat(np.arange(self.n_particles), self.n_restframes)

            # remove indices that would lead to diagonal entries for COMBINATIONS boosting
            if self.boost_mode == self.COMBINATIONS:
                no_diag = np.hstack((triu_range(self.n_particles), tril_range(self.n_particles)))
                l_indices = l_indices[no_diag]
                p_indices = p_indices[no_diag]

            # update Lambda and particles
            Lambda = tf.gather(Lambda, l_indices, axis=1)
            particles = tf.gather(particles, p_indices, axis=1)

        # store the final boost matrix
        self.Lambda = Lambda

        # actual boosting
        boosted_particles = tf.matmul(self.Lambda, particles)

        # remove the last dimension resulting from multiplication and save
        self.boosted_particles = tf.squeeze(boosted_particles, axis=-1, name="boosted_particles")

    def build_auxiliary(self):
        """
        Build combinations of auxiliary input features using the same approach as for particles and
        restframes.
        """
        if self.n_aux <= 0:
            raise Exception("cannot build auxiliary features when n_aux is not positive")

        # build the features via a simple matmul, mapped over the last axis
        self.aux_features = tf.concat([
            tf.matmul(self.inputs_aux[..., i], self.aux_weights[..., i])
            for i in range(self.n_aux)
        ], axis=1)

    def build_features(self, features=("E", "px", "py", "pz"), external_features=None):
        """
        Builds the output features. *features* should be a list of feature names as registered to
        the :py:attr:`feature_factory` instance. When *None*, the default features
        ``["E", "px", "py", "pz"]`` are built. *external_features* can be a list of tensors of
        externally produced features, that are concatenated with the built features.
        """
        symbolic = _is_symbolic(self.inputs)

        # clear the feature caches
        self.feature_factory.clear_caches()

        # create the list of feature ops to concat
        concat = []
        for name in features:
            func = getattr(self.feature_factory, name)
            if func is None:
                raise ValueError("unknown feature '{}'".format(name))
            concat.append(func(_symbolic=symbolic))

        # save intermediate boosted features
        self.boosted_features = tf.concat(concat, axis=-1)

        # add auxiliary features
        if self.n_aux > 0:
            concat.append(self.aux_features)

        # add external features
        if external_features is not None:
            if isinstance(external_features, (list, tuple)):
                concat.extend(list(external_features))
            else:
                concat.append(external_features)

        # save combined features
        self.features = tf.concat(concat, axis=-1)


def _is_symbolic(t):
    """
    Returs *True* when a tensor *t* is a symbolic tensor.
    """
    if len(t.shape) > 0 and t.shape[0] is None:
        return True
    elif callable(getattr(tf_ops, "_is_keras_symbolic_tensor", None)) and \
            tf_ops._is_keras_symbolic_tensor(t):
        return True
    elif getattr(tf_ops, "EagerTensor", None) is not None and isinstance(t, tf_ops.EagerTensor):
        return False
    elif callable(getattr(t, "numpy", None)):
        return False
    else:
        # no other check to perform, assume it is eager
        return False


class FeatureFactoryBase(object):
    """
    Base class of the feature factory. It does not implement actual features but rather the
    feature wrapping and tensor caching functionality. So-called hidden features are also subject to
    caching but are not supposed to be accessed by the LBN. They rather provide intermediate results
    that are used in multiple places and retained for performance purposes.
    """

    DISABLE_CACHE = False

    @classmethod
    def feature(cls, shape_func, hidden=False):
        def decorator(func):
            name = func.__name__

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                no_cache = kwargs.get("_no_cache", self.DISABLE_CACHE)
                symbolic = kwargs.get("_symbolic", False)

                # get the result of the wrapped feature, with or without caching
                if no_cache:
                    return tf.identity(func(self, *args, **kwargs), name=name)
                else:
                    cache = self._symbolic_tensor_cache if symbolic else self._eager_tensor_cache
                    if name not in cache:
                        cache[name] = tf.identity(func(self, *args, **kwargs), name=name)
                    return cache[name]

            # store attributes on the feature wrapper for later use
            wrapper._feature = True
            wrapper._func = func
            wrapper._shape_func = shape_func
            wrapper._hidden = hidden

            return wrapper

        return decorator

    @classmethod
    def hidden_feature(cls, func):
        return cls.feature(None, hidden=True)(func)

    @classmethod
    def single_feature(cls, func):
        shape_func = lambda n_out: n_out
        return cls.feature(shape_func)(func)

    @classmethod
    def pair_feature(cls, func):
        shape_func = lambda n_out: (n_out**2 - n_out) / 2
        return cls.feature(shape_func)(func)

    def __init__(self, lbn):
        super(FeatureFactoryBase, self).__init__()

        # reference to the lbn instance
        self.lbn = lbn

        # some shorthands
        self.n = lbn.n_out
        self.epsilon = lbn.epsilon

        # cached symbolic tensors stored by name
        self._symbolic_tensor_cache = {}

        # cached eager tensors stored by name
        self._eager_tensor_cache = {}

        # dict of registered feature functions without hidden ones
        self._feature_funcs = {}
        for attr in dir(self):
            func = getattr(self, attr)
            if getattr(func, "_feature", False) and not func._hidden:
                self._feature_funcs[attr] = func

    def clear_symbolic_cache(self):
        """
        Clears the current eager tensor cache.
        """
        self._symbolic_tensor_cache.clear()

    def clear_eager_cache(self):
        """
        Clears the current eager tensor cache.
        """
        self._eager_tensor_cache.clear()

    def clear_caches(self):
        """
        Clears both the current eager and symbolic tensor caches.
        """
        self.clear_symbolic_cache()
        self.clear_eager_cache()


class FeatureFactory(FeatureFactoryBase):
    """
    Default feature factory implementing various generic feature mappings.
    """

    def __init__(self, lbn):
        super(FeatureFactory, self).__init__(lbn)

        # pairwise features are computed by multiplying row and column vectors to obtain a
        # matrix from which we want to extract the values of the upper triangle w/o diagonal,
        # so store these upper triangle indices for later use in tf.gather
        self.triu_indices = triu_range(self.n)

    @FeatureFactoryBase.single_feature
    def E(self, **opts):
        """
        Energy.
        """
        return self.lbn.boosted_particles[..., 0]

    @FeatureFactoryBase.single_feature
    def px(self, **opts):
        """
        Momentum component x.
        """
        return self.lbn.boosted_particles[..., 1]

    @FeatureFactoryBase.single_feature
    def py(self, **opts):
        """
        Momentum component y.
        """
        return self.lbn.boosted_particles[..., 2]

    @FeatureFactoryBase.single_feature
    def pz(self, **opts):
        """
        Momentum component z.
        """
        return self.lbn.boosted_particles[..., 3]

    @FeatureFactoryBase.hidden_feature
    def _pvec(self, **opts):
        """
        Momentum vector. Hidden.
        """
        return self.lbn.boosted_particles[..., 1:]

    @FeatureFactoryBase.hidden_feature
    def _p2(self, **opts):
        """
        Squared absolute momentum. Hidden.
        """
        return tf.maximum(tf.reduce_sum(self._pvec(**opts)**2, axis=-1), self.epsilon)

    @FeatureFactoryBase.single_feature
    def p(self, **opts):
        """
        Absolute momentum.
        """
        return self._p2(**opts)**0.5

    @FeatureFactoryBase.single_feature
    def pt(self, **opts):
        """
        Scalar, transverse momentum.
        """
        return tf.maximum(self._p2(**opts) - self.pz(**opts)**2, self.epsilon)**0.5

    @FeatureFactoryBase.single_feature
    def eta(self, **opts):
        """
        Pseudorapidity.
        """
        return tf.atanh(tf.clip_by_value(self.pz(**opts) / self.p(**opts),
            self.epsilon - 1, 1 - self.epsilon))

    @FeatureFactoryBase.single_feature
    def phi(self, **opts):
        """
        Azimuth.
        """
        return tf.atan2(tf_non_zero(self.py(**opts), self.epsilon), self.px(**opts))

    @FeatureFactoryBase.single_feature
    def m(self, **opts):
        """
        Mass.
        """
        return tf.maximum(self.E(**opts)**2 - self._p2(**opts), self.epsilon)**0.5

    @FeatureFactoryBase.single_feature
    def beta(self, **opts):
        """
        Relativistic speed, v/c or p/E.
        """
        return self.p(**opts) / tf.maximum(self.E(**opts), self.epsilon)

    @FeatureFactoryBase.single_feature
    def gamma(self, **opts):
        """
        Relativistic gamma factor, 1 / sqrt(1-beta**2) or E / m.
        """
        return self.E(**opts) / tf.maximum(self.m(**opts), self.epsilon)

    @FeatureFactoryBase.pair_feature
    def pair_dr(self, **opts):
        """
        Distance between all pairs of particles in the eta-phi plane.
        """
        # eta difference on lower triangle elements
        d_eta = tf.reshape(self.eta(**opts), (-1, self.n, 1)) - tf.reshape(self.eta(**opts),
            (-1, 1, self.n))
        d_eta = tf.gather(tf.reshape(d_eta, (-1, self.n**2)), self.triu_indices, axis=1)

        # phi difference on lower triangle elements, handle boundaries
        d_phi = tf.reshape(self.phi(**opts), (-1, self.n, 1)) - tf.reshape(self.phi(**opts),
            (-1, 1, self.n))
        d_phi = tf.gather(tf.reshape(d_phi, (-1, self.n**2)), self.triu_indices, axis=1)
        d_phi = tf.abs(d_phi)
        d_phi = tf.minimum(d_phi, 2. * np.math.pi - d_phi)

        return (d_eta**2 + d_phi**2)**0.5

    @FeatureFactoryBase.hidden_feature
    def _pvec_norm(self, **opts):
        """
        Normalized momentum vector. Hidden.
        """
        return self._pvec(**opts) / tf.expand_dims(self.p(**opts), axis=-1)

    @FeatureFactoryBase.hidden_feature
    def _pvec_norm_T(self, **opts):
        """
        Normalized, transposed momentum vector. Hidden.
        """
        return tf.transpose(self._pvec_norm(**opts), perm=[0, 2, 1])

    @FeatureFactoryBase.pair_feature
    def pair_cos(self, **opts):
        """
        Cosine of the angle between all pairs of particles.
        """
        # cos = (p1 x p2) / (|p1| x |p2|) = (p1 / |p1|) x (p2 / |p2|)
        all_pair_cos = tf.matmul(self._pvec_norm(**opts), self._pvec_norm_T(**opts))

        # return only upper triangle without diagonal
        return tf.gather(tf.reshape(all_pair_cos, [-1, self.n**2]), self.triu_indices, axis=1)

    @FeatureFactoryBase.pair_feature
    def pair_ds(self, **opts):
        """
        Sign-conserving Minkowski space distance between all pairs of particles.
        """
        # (dE**2 - dpx**2 - dpy**2 - dpz**2)**0.5
        # first, determine all 4-vector differences
        pvm = tf.expand_dims(self.lbn.boosted_particles, axis=-2)
        pvm_T = tf.transpose(pvm, perm=[0, 2, 1, 3])
        all_diffs = pvm - pvm_T

        # extract elements of the upper triangle w/o diagonal and calculate their norm
        diffs = tf.gather(tf.reshape(all_diffs, [-1, self.n**2, 4]), self.triu_indices, axis=1)
        diffs_E = diffs[..., 0]
        diffs_p2 = tf.reduce_sum(diffs[..., 1:]**2, axis=-1)

        ds = diffs_E**2 - diffs_p2
        return tf.sign(ds) * tf.abs(ds)**0.5

    @FeatureFactoryBase.pair_feature
    def pair_dy(self, **opts):
        """
        Rapidity difference between all pairs of particles.
        """
        # dy = y1 - y2 = atanh(beta1) - atanh(beta2)
        beta = tf.clip_by_value(self.beta(**opts), self.epsilon, 1 - self.epsilon)
        dy = tf.atanh(tf.expand_dims(beta, axis=-1)) - tf.atanh(tf.expand_dims(beta, axis=-2))

        # return only upper triangle without diagonal
        return tf.gather(tf.reshape(dy, [-1, self.n**2]), self.triu_indices, axis=1)


def tf_non_zero(t, epsilon):
    """
    Ensures that all zeros in a tensor *t* are replaced by *epsilon*.
    """
    # use combination of abs and sign instead of a where op
    return t + (1 - tf.abs(tf.sign(t))) * epsilon


def tril_range(n, k=-1):
    """
    Returns a 1D numpy array containing all lower triangle indices of a square matrix with size *n*.
    *k* is the offset from the diagonal.
    """
    tril_indices = np.tril_indices(n, k)
    return np.arange(n**2).reshape(n, n)[tril_indices]


def triu_range(n, k=1):
    """
    Returns a 1D numpy array containing all upper triangle indices of a square matrix with size *n*.
    *k* is the offset from the diagonal.
    """
    triu_indices = np.triu_indices(n, k)
    return np.arange(n**2).reshape(n, n)[triu_indices]


class LBNLayer(tf.keras.layers.Layer):
    """
    Keras layer of the :py:class:`LBN` that forwards the standard interface of :py:meth:`__init__`
    and py:meth:`__call__`.

    .. py:attribute:: lbn
       type: LBN

       Reference to the internal :py:class:`LBN` instance that is initialized with the contructor
       arguments of this class.
    """

    def __init__(self, input_shape, *args, **kwargs):
        # store and remove kwargs that are not passed to the LBN but to the layer init
        layer_kwargs = {
            "input_shape": input_shape,
            "dtype": kwargs.pop("dtype", None),
            "dynamic": kwargs.pop("dynamic", False),
        }
        # for whatever reason, keras calls this contructor again
        # with batch_input_shape set when input_shape was accepted
        if "batch_input_shape" in kwargs:
            layer_kwargs["batch_input_shape"] = kwargs.pop("batch_input_shape")

        # store names of features to build
        self._features = kwargs.pop("features", None)

        # store external features to concatenate with the lbn outputs
        self._external_features = kwargs.pop("external_features", None)

        # create the LBN instance with the remaining arguments
        self.lbn = LBN(*args, **kwargs)

        # the input_shape is mandatory so we can build right away
        self.build(input_shape)

        # layer init
        super(LBNLayer, self).__init__(name=self.lbn.name, trainable=self.lbn.trainable,
            **layer_kwargs)

    def build(self, input_shape):
        # build the lbn
        self.lbn.build(input_shape, features=self._features,
            external_features=self._external_features)

        # store references to the trainable weights
        # (not necessarily the weights used in combinations)
        self.particle_weights = self.lbn.particle_weights
        self.restframe_weights = self.lbn.restframe_weights
        self.aux_weights = self.lbn.aux_weights

        super(LBNLayer, self).build(input_shape)

    def call(self, inputs):
        return self.lbn(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.lbn.n_features)

    def get_config(self):
        config = super(LBNLayer, self).get_config()
        config.update({
            "input_shape": (self.lbn.n_in, self.lbn.n_dim),
            "n_particles": self.lbn.n_particles,
            "n_restframes": self.lbn.n_restframes,
            "n_auxiliaries": self.lbn.n_auxiliaries,
            "boost_mode": self.lbn.boost_mode,
            "abs_particle_weights": self.lbn.abs_particle_weights,
            "clip_particle_weights": self.lbn.clip_particle_weights,
            "abs_restframe_weights": self.lbn.abs_restframe_weights,
            "clip_restframe_weights": self.lbn.clip_restframe_weights,
            "epsilon": self.lbn.epsilon,
            "seed": self.lbn.seed,
            "features": self._features,
        })
        return config
