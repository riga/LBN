# coding: utf-8

"""
TensorFlow implementation of the Lorentz Boost Network (LBN). https://arxiv.org/abs/1812.09722.
"""


__author__ = "Marcel Rieger"
__copyright__ = "Copyright 2018-2019, Marcel Rieger"
__license__ = "BSD"
__credits__ = ["Martin Erdmann", "Erik Geiser", "Yannik Rath", "Marcel Rieger"]
__contact__ = "https://git.rwth-aachen.de/3pia/lbn"
__email__ = "marcel.rieger@cern.ch"
__version__ = "1.0.5"

__all__ = ["LBN", "LBNLayer", "FeatureFactoryBase", "FeatureFactory"]


import functools

import numpy as np
import tensorflow as tf


# tf version flags
TF1 = tf.__version__.startswith("1.")
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
    stability. *name* is the main namespace of the LBN and defaults to the class name.

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

    Instances of this class store most of the intermediate tensors (such as inputs, combinations
    weights, boosted particles, boost matrices, raw features, etc) for later inspection. Note that
    most of these tensors are set after :py:meth:`build` (or the :py:meth:`__call__` shorthand as
    shown above) are invoked.
    """

    # available boost modes
    PAIRS = "pairs"
    PRODUCT = "product"
    COMBINATIONS = "combinations"

    def __init__(self, n_particles, n_restframes=None, boost_mode=PAIRS, feature_factory=None,
            particle_weights=None, abs_particle_weights=True, clip_particle_weights=False,
            restframe_weights=None, abs_restframe_weights=True, clip_restframe_weights=False,
            weight_init=None, epsilon=1e-5, name=None):
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

        # particle weights and settings
        self.particle_weights = particle_weights
        self.abs_particle_weights = abs_particle_weights
        self.clip_particle_weights = clip_particle_weights

        # rest frame weigths and settings
        self.restframe_weights = restframe_weights
        self.abs_restframe_weights = abs_restframe_weights
        self.clip_restframe_weights = clip_restframe_weights

        # custom weight init parameters in a tuple (mean, stddev)
        self.weight_init = weight_init

        # epsilon for numerical stability
        self.epsilon = epsilon

        # internal name
        self.name = name or self.__class__.__name__

        # sizes that are set during build
        self.n_in = None  # number of input particles
        self.n_dim = None  # size per input vector, must be four

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

        # Lorentz boost matrix with shape (batch, n_out, 4, 4)
        self.Lambda = None

        # boosted particles with shape (batch, n_out, 4)
        self.boosted_particles = None

        # intermediate features
        self._raw_features = None  # raw features before batch normalization, etc

        # final output features
        self.features = None

        # initialize the feature factory
        if feature_factory is None:
            feature_factory = FeatureFactory
        elif not issubclass(feature_factory, FeatureFactoryBase):
            raise TypeError("feature_factory '{}' is not a subclass of FeatureFactoryBase".format(
                feature_factory))
        self.feature_factory = feature_factory(self)

    @property
    def available_features(self):
        """
        Shorthand to access the list of available features in the :py:attr:`feature_factory`.
        """
        return list(self.feature_factory._feature_funcs.keys())

    @property
    def n_features(self):
        """
        Returns the number of created output features which depends on the number of boosted
        particles and the feature set.
        """
        if self.features is None:
            return None

        return int(self.features.shape[-1])

    def register_feature(self, func=None, **kwargs):
        """
        Shorthand to register a new feautre to the current :py:attr:`feature_factory` instance. Can
        be used as a (configurable) decorator. The decorated function receives the feature factory
        instance as the only argument. All *kwargs* are forwarded to
        :py:meth:`FeatureFactoryBase._wrap_feature`. Example:

        .. code-block:: python

            lbn = LBN(10, boost_mode=LBN.PAIRS)

            @lbn.register_feature
            def px_plus_py(ff):
                return ff.px() + ff.py()

            print("px_plus_py" in lbn.available_features)  # -> True

            # or register with a different name
            @lbn.register_feature(name="pxy")
            def px_plus_py(ff):
                return ff.px() + ff.py()

            print("pxy" in lbn.available_features)  # -> True
        """
        def decorator(func):
            return self.feature_factory._wrap_feature(func, **kwargs)

        return decorator(func) if func else decorator

    def __call__(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`build`.
        """
        return self.build(*args, **kwargs)

    def build(self, inputs, **kwargs):
        """
        Builds the LBN structure layer by layer within dedicated variable scopes. *input* must be a
        tensor of the input four-vectors. All *kwargs* are forwarded to :py:meth:`build_features`.
        """
        with tf.name_scope(self.name):
            with tf.name_scope("inputs"):
                self.handle_input(inputs)

            with tf.name_scope("constants"):
                self.build_constants()

            with tf.name_scope("particles"):
                self.build_combinations("particle", self.n_particles)

            # rest frames are not built for COMBINATIONS boost mode
            if self.boost_mode != self.COMBINATIONS:
                with tf.name_scope("restframes"):
                    self.build_combinations("restframe", self.n_restframes)

            with tf.name_scope("boost"):
                self.build_boost()

            with tf.name_scope("features"):
                self.build_features(**kwargs)
            self.features = self._raw_features

        return self.features

    def handle_input(self, inputs):
        """
        Takes the passed four-vector *inputs* and infers dimensions and some internal tensors.
        """
        # inputs are expected to be four-vectors with the shape (batch, n_in, 4)
        # convert them in case they have the shape (batch, 4 * n_in)
        if len(inputs.shape) == 2 and inputs.shape[-1] % 4 == 0:
            inputs = tf.reshape(inputs, (-1, inputs.shape[-1] // 4, 4))

        # store the input vectors
        self.inputs = inputs

        # infer sizes
        self.n_in = int(self.inputs.shape[1])
        self.n_dim = int(self.inputs.shape[2])
        if self.n_dim != 4:
            raise Exception("input dimension must be 4 to represent 4-vectors")

        # split 4-vector components
        names = ["E", "px", "py", "pz"]
        split = [1, 1, 1, 1]
        for t, name in zip(tf.split(self.inputs, split, axis=-1), names):
            setattr(self, "inputs_" + name, tf.squeeze(t, -1))

    def build_constants(self):
        """
        Builds the internal constants for the boost matrix.
        """
        # 4x4 identity
        self.I = tf.constant(np.identity(4), tf.float32)

        # U matrix
        self.U = tf.constant([[-1, 0, 0, 0]] + 3 * [[0, -1, -1, -1]], tf.float32)

    def build_combinations(self, name, m):
        """
        Builds the combination layers which are quite similiar for particles and rest frames. Hence,
        *name* must be either ``"particle"`` or ``"restframe"``, and *m* is the corresponding number
        of combinations.
        """
        if name not in ("particle", "restframe"):
            raise ValueError("unknown combination name '{}'".format(name))

        # determine the weight tensor shape
        weight_shape = (self.n_in, m)

        # name helper
        name_ = lambda tmpl: tmpl.format(name)

        # build the weight matrix, or check it if already set
        W = getattr(self, name_("{}_weights"))
        if W is None:
            # build a new weight matrix
            if isinstance(self.weight_init, tuple):
                mean, stddev = self.weight_init
            else:
                mean, stddev = 0., 1. / m

            W = tf.Variable(tf.random.normal(weight_shape, mean, stddev, dtype=tf.float32))

        else:
            # W is set externally, check the shape, consider batching
            shape = tuple(W.shape.as_list())
            if shape != weight_shape:
                raise ValueError("external {}_weights shape {} does not match {}".format(
                    name, shape, weight_shape))

        # store as raw weights before applying abs or clipping
        W = tf.identity(W, name=name_("raw_{}_weights"))

        # apply abs
        if getattr(self, name_("abs_{}_weights")):
            W = tf.abs(W, name=name_("abs_{}_weights"))

        # apply clipping
        clip = getattr(self, name_("clip_{}_weights"))
        if clip is True:
            clip = self.epsilon
        if clip is not False:
            W = tf.maximum(W, clip, name=name_("clipped_{}_weights"))

        # assign a name to the final weights
        W = tf.identity(W, name=name_("{}_weights"))

        # create four-vectors of combinations
        E = tf.matmul(self.inputs_E, W, name=name_("{}s_E"))
        px = tf.matmul(self.inputs_px, W, name=name_("{}s_px"))
        py = tf.matmul(self.inputs_py, W, name=name_("{}s_py"))
        pz = tf.matmul(self.inputs_pz, W, name=name_("{}s_pz"))

        # create the full 3- and 4-vector stacks again
        p = tf.stack([px, py, pz], axis=-1, name=name_("{}s_pvec"))
        q = tf.stack([E, px, py, pz], axis=-1, name=name_("{}s"))

        # save all tensors for later inspection
        setattr(self, name_("{}_weights"), W)
        setattr(self, name_("{}s_E"), E)
        setattr(self, name_("{}s_px"), px)
        setattr(self, name_("{}s_py"), py)
        setattr(self, name_("{}s_pz"), pz)
        setattr(self, name_("{}s_pvec"), p)
        setattr(self, name_("{}s"), q)

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

    def build_features(self, features=None, external_features=None):
        """
        Builds the output features. *features* should be a list of feature names as registered to
        the :py:attr:`feature_factory` instance. When *None*, the default features
        ``["E", "px", "py", "pz"]`` are built. *external_features* can be a list of tensors of
        externally produced features, that are concatenated to the built features.
        """
        # default to reshaped 4-vector elements
        if features is None:
            features = ["E", "px", "py", "pz"]

        # create the list of feature ops to concat
        concat = []
        for name in features:
            func = getattr(self.feature_factory, name)
            if func is None:
                raise ValueError("unknown feature '{}'".format(name))
            concat.append(func())

        # add external features
        if external_features is not None:
            if isinstance(external_features, (list, tuple)):
                concat.extend(list(external_features))
            else:
                concat.append(external_features)

        # save raw features
        self._raw_features = tf.concat(concat, axis=-1)


class LBNLayer(tf.keras.layers.Layer):
    """
    Keras layer of the :py:class:`LBN` that forwards the standard interface of :py:meth:`__init__`
    and py:meth:`__call__`.

    .. py:attribute:: lbn
       type: LBN

       Reference to the internal :py:class:`LBN` instance that is initialized with the contructor
       arguments of this class.
    """

    def __init__(self, *args, **kwargs):
        super(LBNLayer, self).__init__()

        # store names of features to build
        self.feature_names = kwargs.pop("features", None)

        # store the seed
        self.seed = kwargs.pop("seed", None)

        # create the LBN instance with the remaining arguments
        self.lbn = LBN(*args, **kwargs)

    def build(self, input_shape):
        # get the number of input vectors
        n_in = input_shape[-2] if len(input_shape) == 3 else input_shape[-1] // 4

        def add_weight(name, m):
            if isinstance(self.lbn.weight_init, tuple):
                mean, stddev = self.lbn.weight_init
            else:
                mean, stddev = 0., 1. / m

            weight_shape = (n_in, m)
            weight_name = "{}_weights".format(name)
            weight_init = tf.keras.initializers.RandomNormal(mean=mean, stddev=stddev,
                seed=self.seed)

            W = self.add_weight(name=weight_name, shape=weight_shape, initializer=weight_init,
                trainable=True)

            setattr(self, weight_name, W)
            setattr(self.lbn, weight_name, W)

        # add weights, depending on the boost mode
        add_weight("particle", self.lbn.n_particles)
        if self.lbn.boost_mode != LBN.COMBINATIONS:
            add_weight("restframe", self.lbn.n_restframes)

        super(LBNLayer, self).build(input_shape)

    if TF2:
        @tf.function
        def call(self, inputs):
            # forward to lbn.__call__
            return self.lbn(inputs, features=self.feature_names)
    else:
        def call(self, inputs):
            # forward to lbn.__call__
            return self.lbn(inputs, features=self.feature_names)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.lbn.n_features)


class FeatureFactoryBase(object):
    """
    Base class of the feature factory. It does not implement actual features but rather the
    feature wrapping and tensor caching functionality. So-called hidden features are also subject to
    caching but are not supposed to be accessed by the LBN. They rather provide intermediate results
    that are used in multiple places and retained for performance purposes.
    """

    excluded_attributes = ["_wrap_feature", "_wrap_features", "lbn"]

    def __init__(self, lbn):
        super(FeatureFactoryBase, self).__init__()

        # cached tensors stored by name
        # contains also hidden features
        self._tensor_cache = {}

        # dict of registered, bound feature functions
        # does not contain hidden features
        self._feature_funcs = {}

        # wrap all features defined in this class
        self._wrap_features()

        # reference to the lbn instance
        self.lbn = lbn

        # some shorthands
        self.n = lbn.n_out
        self.epsilon = lbn.epsilon

    def _wrap_feature(self, func, name=None, hidden=None):
        """
        Wraps and registers a feature function. It ensures that the stored function is bound to this
        instance. *name* defaults to the actual function name. When *hidden* is *None*, the decision
        is inferred from whether *name* starts with an underscore.
        """
        if not name:
            name = func.__name__
        if hidden is None:
            hidden = name.startswith("_")

        # bind it to self if not bound yet
        if getattr(func, "__self__", None) is None:
            func = func.__get__(self)

        @functools.wraps(func)
        def wrapper(ff, *args, **kwargs):
            if kwargs.pop("no_cache", False):
                return func(*args, **kwargs)
            else:
                if name not in self._tensor_cache:
                    self._tensor_cache[name] = tf.identity(func(*args, **kwargs), name=name)
                return self._tensor_cache[name]

        # register the bound, caching-aware wrapper to this instance
        setattr(self, name, wrapper.__get__(self))

        # store in known feature func if not hidden
        if not hidden:
            self._feature_funcs[name] = wrapper

        return wrapper

    def _wrap_features(self):
        """
        Interprets all non-excluded instance methods as feature functions and replaces them by
        caching-aware wrappers using :py:meth:`_wrap_feature`.
        """
        # loop through attributes
        for name in dir(self):
            # magic method or excluded?
            if name.startswith("__") or name in self.excluded_attributes:
                continue

            # callable?
            func = getattr(self, name)
            if not callable(func):
                continue

            # wrap it
            self._wrap_feature(func, name)


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

    def E(self):
        """
        Energy.
        """
        return self.lbn.boosted_particles[..., 0]

    def px(self):
        """
        Momentum component x.
        """
        return self.lbn.boosted_particles[..., 1]

    def py(self):
        """
        Momentum component y.
        """
        return self.lbn.boosted_particles[..., 2]

    def pz(self):
        """
        Momentum component z.
        """
        return self.lbn.boosted_particles[..., 3]

    def _pvec(self):
        """
        Momentum vector. Hidden.
        """
        return self.lbn.boosted_particles[..., 1:]

    def _p2(self):
        """
        Squared absolute momentum. Hidden.
        """
        return tf.maximum(tf.reduce_sum(self._pvec()**2, axis=-1), self.epsilon)

    def p(self):
        """
        Absolute momentum.
        """
        return self._p2()**0.5

    def pt(self):
        """
        Scalar, transverse momentum.
        """
        return tf.maximum(self._p2() - self.pz()**2, self.epsilon)**0.5

    def eta(self):
        """
        Pseudorapidity.
        """
        return tf.atanh(tf.clip_by_value(self.pz() / self.p(), self.epsilon - 1, 1 - self.epsilon))

    def phi(self):
        """
        Azimuth.
        """
        return tf.atan2(tf_non_zero(self.py(), self.epsilon), self.px())

    def m(self):
        """
        Mass.
        """
        return tf.maximum(self.E()**2 - self._p2(), self.epsilon)**0.5

    def beta(self):
        """
        Relativistic speed, v/c or p/E.
        """
        return self.p() / tf.maximum(self.E(), self.epsilon)

    def gamma(self):
        """
        Relativistic gamma factor, 1 / sqrt(1-beta**2) or E / m.
        """
        return self.E() / tf.maximum(self.m(), self.epsilon)

    def pair_dr(self):
        """
        Distance between all pairs of particles in the eta-phi plane.
        """
        # eta difference on lower triangle elements
        d_eta = tf.reshape(self.eta(), (-1, self.n, 1)) - tf.reshape(self.eta(), (-1, 1, self.n))
        d_eta = tf.gather(tf.reshape(d_eta, (-1, self.n**2)), self.triu_indices, axis=1)

        # phi difference on lower triangle elements, handle boundaries
        d_phi = tf.reshape(self.phi(), (-1, self.n, 1)) - tf.reshape(self.phi(), (-1, 1, self.n))
        d_phi = tf.gather(tf.reshape(d_phi, (-1, self.n**2)), self.triu_indices, axis=1)
        d_phi = tf.abs(d_phi)
        d_phi = tf.minimum(d_phi, 2. * np.math.pi - d_phi)

        return (d_eta**2 + d_phi**2)**0.5

    def _pvec_norm(self):
        """
        Normalized momentum vector. Hidden.
        """
        return self._pvec() / tf.expand_dims(self.p(), axis=-1)

    def _pvec_norm_T(self):
        """
        Normalized, transposed momentum vector. Hidden.
        """
        return tf.transpose(self._pvec_norm(), perm=[0, 2, 1])

    def pair_cos(self):
        """
        Cosine of the angle between all pairs of particles.
        """
        # cos = (p1 x p2) / (|p1| x |p2|) = (p1 / |p1|) x (p2 / |p2|)
        all_pair_cos = tf.matmul(self._pvec_norm(), self._pvec_norm_T())

        # return only upper triangle without diagonal
        return tf.gather(tf.reshape(all_pair_cos, [-1, self.n**2]), self.triu_indices, axis=1)

    def pair_ds(self):
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

    def pair_dy(self):
        """
        Rapidity difference between all pairs of particles.
        """
        # dy = y1 - y2 = atanh(beta1) - atanh(beta2)
        beta = tf.clip_by_value(self.beta(), self.epsilon, 1 - self.epsilon)
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
