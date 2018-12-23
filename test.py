# coding: utf-8

"""
LBN unit tests.
"""


__all__ = ["TestCase"]


import unittest

import numpy as np
import tensorflow as tf

from lbn import LBN, FeatureFactory


tf.enable_eager_execution()


class TestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestCase, self).__init__(*args, **kwargs)

        # create some four-vectors with fixed seed and batch size 2
        self.vectors = create_four_vectors((2, 10), seed=123)
        self.vectors_t = tf.constant(self.vectors, dtype=tf.float32)

        # common feature set
        self.feature_set = ["E", "pt", "eta", "phi", "m", "pair_cos"]

        # custom weights
        self.custom_particle_weights = tf.constant([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] +
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] +
            80 * [0],

        ], shape=[10, 10], dtype=tf.float32)
        self.custom_restframe_weights = tf.constant([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] +
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] +
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] +
            70 * [0],

        ], shape=[10, 10], dtype=tf.float32)

    def test_vectors_seed(self):
        self.assertAlmostEqual(np.sum(self.vectors), 9500.47157362)

    def test_constructor(self):
        lbn = LBN(10)
        self.assertIsInstance(lbn, LBN)

    def test_constructor_boost_mode_pairs(self):
        lbn = LBN(10, boost_mode=LBN.PAIRS, is_training=True)
        self.assertEqual(lbn.n_particles, 10)
        self.assertEqual(lbn.n_restframes, 10)
        self.assertEqual(lbn.n_out, 10)
        self.assertIsNone(lbn.n_features)

        features = lbn(self.vectors_t, features=self.feature_set).numpy()

        self.assertEqual(lbn.n_in, 10)
        self.assertEqual(features.shape[1], lbn.n_features)
        self.assertEqual(features.shape, (2, 95))

    def test_constructor_boost_mode_product(self):
        lbn = LBN(10, 4, boost_mode=LBN.PRODUCT, is_training=True)
        self.assertEqual(lbn.n_particles, 10)
        self.assertEqual(lbn.n_restframes, 4)
        self.assertEqual(lbn.n_out, 40)
        self.assertIsNone(lbn.n_features)

        features = lbn(self.vectors_t, features=self.feature_set).numpy()

        self.assertEqual(lbn.n_in, 10)
        self.assertEqual(features.shape[1], lbn.n_features)
        self.assertEqual(features.shape, (2, 980))

    def test_constructor_boost_mode_combinations(self):
        lbn = LBN(10, boost_mode=LBN.COMBINATIONS, is_training=True)
        self.assertEqual(lbn.n_particles, 10)
        self.assertEqual(lbn.n_restframes, 10)
        self.assertEqual(lbn.n_out, 90)
        self.assertIsNone(lbn.n_features)

        features = lbn(self.vectors_t, features=self.feature_set).numpy()

        self.assertEqual(lbn.n_in, 10)
        self.assertEqual(features.shape[1], lbn.n_features)
        self.assertEqual(features.shape, (2, 4455))

    def test_unknown_boost_mode(self):
        with self.assertRaises(ValueError):
            LBN(10, boost_mode="definitely_not_there", is_training=True)

    def test_pre_build_attributes(self):
        lbn = LBN(10, boost_mode=LBN.PAIRS, is_training=True)

        attrs = ["is_training", "epsilon", "name"]
        for attr in attrs:
            self.assertIsNotNone(getattr(lbn, attr))

    def test_post_build_attributes(self):
        lbn = LBN(10, boost_mode=LBN.PAIRS, is_training=True)
        lbn(self.vectors_t, features=self.feature_set).numpy()

        attrs = [
            "particle_weights", "abs_particle_weights", "clip_particle_weights",
            "restframe_weights", "abs_restframe_weights", "clip_restframe_weights", "n_in", "n_dim",
            "I", "U", "inputs", "inputs_E", "inputs_px", "inputs_py", "inputs_pz", "particles_E",
            "particles_px", "particles_py", "particles_pz", "particles_pvec", "particles",
            "restframes_E", "restframes_px", "restframes_py", "restframes_pz", "restframes_pvec",
            "restframes", "Lambda", "boosted_particles", "_raw_features", "_norm_features",
            "features",
        ]
        for attr in attrs:
            self.assertIsNotNone(getattr(lbn, attr), None)

    def test_batch_norm(self):
        lbn = LBN(10, boost_mode=LBN.PAIRS, batch_norm=True, is_training=True)
        self.assertTrue(lbn.batch_norm_center)
        self.assertTrue(lbn.batch_norm_scale)

        lbn = LBN(10, boost_mode=LBN.PAIRS, batch_norm=(True, False), is_training=True)
        self.assertTrue(lbn.batch_norm_center)
        self.assertFalse(lbn.batch_norm_scale)

        lbn = LBN(10, boost_mode=LBN.PAIRS, batch_norm=(0.5, 2.), is_training=True)
        self.assertEqual(lbn.batch_norm_center, 0.5)
        self.assertEqual(lbn.batch_norm_scale, 2.)

    def test_custom_weights(self):
        lbn = LBN(10, boost_mode=LBN.PAIRS, particle_weights=self.custom_particle_weights,
            restframe_weights=self.custom_restframe_weights, is_training=True)
        lbn(self.vectors_t, features=self.feature_set).numpy()

        self.assertEqual(lbn.particle_weights.numpy().shape, (10, 10))
        self.assertEqual(lbn.restframe_weights.numpy().shape, (10, 10))

        self.assertEqual(np.sum(lbn.particle_weights.numpy()), 3)
        self.assertEqual(np.sum(lbn.restframe_weights.numpy()), 3)

        # compare sum of vector components of first combined particles and restframes in batch pos 1
        target_particle_sum = np.sum(self.vectors[1, 0] + self.vectors[1, 1])
        target_restframe_sum = np.sum(self.vectors[1, 1] + self.vectors[1, 2])

        self.assertAlmostEqual(np.sum(lbn.particles.numpy()[1, 0]), target_particle_sum, 3)
        self.assertAlmostEqual(np.sum(lbn.restframes.numpy()[1, 0]), target_restframe_sum, 3)

        # test wrong shape
        lbn = LBN(10, boost_mode=LBN.PAIRS, particle_weights=self.custom_particle_weights,
            restframe_weights=self.custom_restframe_weights[:-1], is_training=True)
        with self.assertRaises(ValueError):
            lbn(self.vectors_t, features=self.feature_set).numpy()

    def test_boosting_pairs(self):
        lbn = LBN(10, boost_mode=LBN.PAIRS, particle_weights=self.custom_particle_weights,
            restframe_weights=self.custom_restframe_weights, is_training=True)
        lbn(self.vectors_t, features=self.feature_set).numpy()

        # compare all components of the first boosted particle in batch pos 1
        particle = lbn.particles.numpy()[1, 0]
        components = list(self.vectors[1, 0] + self.vectors[1, 1])
        for i, v in enumerate(components):
            self.assertAlmostEqual(particle[i], v, 3)

        restframe = lbn.restframes.numpy()[1, 0]
        components = list(self.vectors[1, 1] + self.vectors[1, 2])
        for i, v in enumerate(components):
            self.assertAlmostEqual(restframe[i], v, 3)

        # boosted values computed ROOT TLorentzVector's via
        # p = TLorentzVector(particle[1], particle[2], particle[3], particle[0])
        # r = TLorentzVector(restframe[1], restframe[2], restframe[3], restframe[0])
        # p = p.Boost(-r.BoostVector())
        boosted = lbn.boosted_particles.numpy()[1, 0]
        components = [295.55554, -49.238365, -57.182686, -13.449585]
        for i, v in enumerate(components):
            self.assertAlmostEqual(boosted[i], v, 5)

    def test_boosting_product(self):
        lbn = LBN(10, 4, boost_mode=LBN.PRODUCT, particle_weights=self.custom_particle_weights,
            restframe_weights=self.custom_restframe_weights[:, :4], is_training=True)
        lbn(self.vectors_t, features=self.feature_set).numpy()

        # compare all components of the first boosted particle in batch pos 1
        # see test_boosting_pairs for manual boost computation
        boosted = lbn.boosted_particles.numpy()[1, 0]
        components = [295.55554, -49.238365, -57.182686, -13.449585]
        for i, v in enumerate(components):
            self.assertAlmostEqual(boosted[i], v, 5)

    def test_boosting_combinations(self):
        lbn = LBN(10, boost_mode=LBN.COMBINATIONS, particle_weights=self.custom_particle_weights,
            is_training=True)
        lbn(self.vectors_t, features=self.feature_set).numpy()

        # compare all components of the first boosted particle in batch pos 1
        # see test_boosting_pairs for manual boost computation
        p1 = lbn.particles.numpy()[1, 0]
        components = list(self.vectors[1, 0] + self.vectors[1, 1])
        for i, v in enumerate(components):
            self.assertAlmostEqual(p1[i], v, 3)

        p2 = lbn.particles.numpy()[1, 1]
        components = list(self.vectors[1, 0])
        for i, v in enumerate(components):
            self.assertAlmostEqual(p2[i], v, 5)

        # boosted particle 0 is p1 boosted into p2
        boosted = lbn.boosted_particles.numpy()[1, 0]
        components = [701.2019, -592.5028, 141.76767, -197.63687]
        for i, v in enumerate(components):
            self.assertAlmostEqual(boosted[i], v, 3)

        # boosted particle 45 is p2 boosted into p1
        boosted = lbn.boosted_particles.numpy()[1, 45]
        components = [141.52478, 85.97396, -95.47604, 14.268608]
        for i, v in enumerate(components):
            self.assertAlmostEqual(boosted[i], v, 3)

    def test_custom_feature_factory(self):
        class MyFeatureFactory(FeatureFactory):

            def px_plus_py(self):
                return self.px() + self.py()

        lbn = LBN(10, boost_mode=LBN.PAIRS, is_training=True, feature_factory=MyFeatureFactory)
        self.assertIn("px_plus_py", lbn.available_features)

        with self.assertRaises(TypeError):
            LBN(10, boost_mode=LBN.PAIRS, is_training=True, feature_factory="foo")

    def test_register_feature(self):
        lbn = LBN(10, boost_mode=LBN.PAIRS, is_training=True)
        self.assertNotIn("px_plus_py", lbn.available_features)

        @lbn.register_feature
        def px_plus_py(factory):
            return factory.px() + factory.py()

        self.assertIn("px_plus_py", lbn.available_features)

    def test_external_features(self):
        lbn = LBN(10, boost_mode=LBN.PAIRS, is_training=True)

        ext = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
        features = lbn(self.vectors_t, features=self.feature_set, external_features=ext).numpy()

        self.assertEqual(features.shape[1], lbn.n_features)
        self.assertEqual(features.shape, (2, 97))

    def test_feature_caching(self):
        class MyFeatureFactory(FeatureFactory):

            def __init__(self, *args, **kwargs):
                super(MyFeatureFactory, self).__init__(*args, **kwargs)
                self.count = 0

            def px_plus_py(self):
                self.count += 1
                return self.px() + self.py()

        lbn = LBN(10, boost_mode=LBN.PAIRS, is_training=True, feature_factory=MyFeatureFactory)
        self.assertEqual(lbn.feature_factory.count, 0)

        lbn(self.vectors_t, features=self.feature_set + ["px_plus_py"]).numpy()
        self.assertEqual(lbn.feature_factory.count, 1)

        lbn.feature_factory.px_plus_py()
        self.assertEqual(lbn.feature_factory.count, 1)

    def test_features(self):
        lbn = LBN(10, boost_mode=LBN.PAIRS, particle_weights=self.custom_particle_weights,
            restframe_weights=self.custom_restframe_weights, is_training=True)

        all_features = [
            "E", "px", "py", "pz", "pt", "p", "m", "phi", "eta", "beta", "gamma", "pair_cos",
            "pair_dr", "pair_ds", "pair_dy",
        ]
        self.assertEqual(set(lbn.available_features), set(all_features))

        # also add a custom feature
        @lbn.register_feature
        def px_plus_py(factory):
            return factory.px() + factory.py()

        lbn(self.vectors_t, features=all_features)

        # make all tests on the first boosted particle at batch pos 1
        self.assertAlmostEqual(lbn.feature_factory.E().numpy()[1, 0], 295.55554, 5)
        self.assertAlmostEqual(lbn.feature_factory.px().numpy()[1, 0], -49.238365, 5)
        self.assertAlmostEqual(lbn.feature_factory.py().numpy()[1, 0], -57.182686, 5)
        self.assertAlmostEqual(lbn.feature_factory.pz().numpy()[1, 0], -13.449585, 5)
        self.assertAlmostEqual(lbn.feature_factory.pt().numpy()[1, 0], 75.460428, 5)
        self.assertAlmostEqual(lbn.feature_factory.p().numpy()[1, 0], 76.649641, 5)
        self.assertAlmostEqual(lbn.feature_factory.m().numpy()[1, 0], 285.443356, 5)
        self.assertAlmostEqual(lbn.feature_factory.phi().numpy()[1, 0], -2.281683, 5)
        self.assertAlmostEqual(lbn.feature_factory.eta().numpy()[1, 0], -0.177303, 5)
        self.assertAlmostEqual(lbn.feature_factory.beta().numpy()[1, 0], 0.259341, 5)
        self.assertAlmostEqual(lbn.feature_factory.gamma().numpy()[1, 0], 1.035426, 5)

        # test pairwise features w.r.t. boosted particle 2, i.e., feature pos 0
        self.assertAlmostEqual(lbn.feature_factory.pair_cos().numpy()[1, 0], 0.58045, 5)
        self.assertAlmostEqual(lbn.feature_factory.pair_dr().numpy()[1, 0], 0.96127, 5)
        self.assertAlmostEqual(lbn.feature_factory.pair_ds().numpy()[1, 0], -457.953327, 4)
        self.assertAlmostEqual(lbn.feature_factory.pair_dy().numpy()[1, 0], -2.745842, 5)

        # test the custom feature
        self.assertAlmostEqual(lbn.feature_factory.px_plus_py().numpy()[1, 0], -106.421051, 5)


def create_four_vectors(n, p_low=10., p_high=100., m_low=0.1, m_high=50., seed=None):
    """
    Creates a numpy array with shape ``n + (4,)`` describing four-vectors of particles whose
    momentum components are uniformly distributed between *p_low* and *p_high*, and masses between
    *m_low* and *m_high*. When *seed* is not *None*, it is initially passed to ``np.random.seed()``.
    """
    if seed is not None:
        np.random.seed(seed)

    # ensure n is a tuple
    if not isinstance(n, tuple):
        n = (n,)

    # create random four-vectors
    vecs = np.abs(np.random.normal(p_low, p_high, n + (4,)))

    # the energy is also random and might be lower than the momentum,
    # make sure that masses are positive and uniformly distributed
    m = np.abs(np.random.normal(m_low, m_high, n))

    # insert into vectors
    p = np.sqrt(np.sum(vecs[..., 1:]**2, axis=-1))
    E = (p**2 + m**2)**0.5
    vecs[..., 0] = E

    return vecs
