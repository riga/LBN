from __future__ import annotations

__author__ = "Marcel Rieger"
__email__ = "github.riga@icloud.com"
__copyright__ = "Copyright 2025, Marcel Rieger"
__credits__ = ["Marcel Rieger"]
__contact__ = "https://github.com/riga/lbn"
__license__ = "BSD-3-Clause"
__status__ = "Development"
__version__ = "1.2.2"
__all__ = ["LBN"]

from typing import Sequence

import torch


class LBN(torch.nn.Module):
    """
    Torch implementation of the LBN (Lorentz Boosted Network) feature extractor.
    For details and nomenclature see https://arxiv.org/pdf/1812.09722.
    """

    KNOWN_FEATURES = [
        "e", "px", "py", "pz",
        "pt", "eta", "phi", "m",
        "pair_cos", "pair_dr",
    ]

    DEFAULT_FEATURES = ["e", "pt", "eta", "phi", "m", "pair_cos"]

    def __init__(
        self,
        N: int,
        M: int,
        *,
        features: Sequence[str] | None = None,
        weight_init_scale: float | int = 1.0,
        clip_weights: bool = False,
        eps: float = 1.0e-5,
    ) -> None:
        super().__init__()

        # validate features
        if features is None:
            features = self.DEFAULT_FEATURES
        for f in features:
            if f not in self.KNOWN_FEATURES:
                raise ValueError(f"unknown feature '{f}', known features are: {self.KNOWN_FEATURES}")

        # store settings
        self.N = N
        self.M = M
        self.features = list(features)
        self.weight_init_scale = weight_init_scale
        self.clip_weights = clip_weights
        self.eps = eps

        # constants
        self.register_buffer("I4", torch.eye(4, dtype=torch.float32))  # (4, 4)
        self.register_buffer("U", torch.tensor([[-1, 0, 0, 0], *(3 * [[0, -1, -1, -1]])], dtype=torch.float32))
        self.register_buffer("U1", self.U + 1)
        self.register_buffer("lower_tril", torch.tril(torch.ones(M, M, dtype=torch.bool), -1))

        # randomly initialized weights for projections
        self.particle_w = torch.nn.Parameter(torch.rand(N, M) * weight_init_scale)
        self.restframe_w = torch.nn.Parameter(torch.rand(N, M) * weight_init_scale)

    def __repr__(self) -> str:
        params = {
            "N": self.N,
            "M": self.M,
            "features": ",".join(self.features),
            "clip": self.clip_weights,
        }
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_str}, {hex(id(self))})"

    @property
    def out_features(self) -> int:
        # determine number of pair-wise feature projections
        n_pair = sum(1 for f in self.features if f.startswith("pair_"))

        # compute output dimension
        n = (
            (len(self.features) - n_pair) * self.M +
            n_pair * (self.M**2 - M) // 2
        )

        return n

    def update_particle_weights(self, w: torch.Tensor) -> torch.Tensor:
        return w

    def update_restframe_weights(self, w: torch.Tensor) -> torch.Tensor:
        return w

    def update_boosted_vectors(self, boosted_vecs: torch.Tensor) -> torch.Tensor:
        return boosted_vecs

    def forward(
        self,
        e: torch.Tensor,
        px: torch.Tensor | None = None,
        py: torch.Tensor | None = None,
        pz: torch.Tensor | None = None,
        /,
    ) -> torch.Tensor:
        # e, px, py, pz: (B, N)
        E, PX, PY, PZ = range(4)

        # handle input
        # all arguments are given, check shapes and stack them, otherwise e must be already stacked
        if (n_missing := [px, py, pz].count(None)) == 3:
            # only e provided
            if e.dim != 3 or tuple(e.shape[1:]) != (4, self.N):
                raise Exception(f"input four-vectors have wrong shape {tuple(e.shape)}, expected (B, 4, N)")
            input_vecs = e  # (B, 4, N)
        elif n_missing == 0:
            # all arguments provided, stack 4-vectors
            input_vecs = torch.stack((e, px, py, pz), dim=1)  # (B, 4, N)
            pass
        else:
            raise Exception(f"forward() expects either 1 or 4 arguments, got {4 - n_missing}")

        # optionally update particle and restframe weights
        particle_w = self.update_particle_weights(self.particle_w)
        restframe_w = self.update_restframe_weights(self.restframe_w)

        # optionally clip weights to prevent them going negative
        if self.clip_weights:
            particle_w = torch.clamp(particle_w, min=0.0)
            restframe_w = torch.clamp(restframe_w, min=0.0)

        # create combinations
        particle_vecs = torch.matmul(input_vecs, particle_w)  # (B, 4, M)
        restframe_vecs = torch.matmul(input_vecs, restframe_w)  # (B, 4, M)
        # transpose to (B, M, 4)
        particle_vecs = particle_vecs.permute(0, 2, 1)
        restframe_vecs = restframe_vecs.permute(0, 2, 1)

        # regularize vectors such that e > p
        particle_p = torch.sum(particle_vecs[..., PX:]**2, dim=-1)**0.5  # (B, M)
        particle_vecs[..., E] = torch.maximum(particle_vecs[..., E], particle_p + self.eps)
        restframe_p = torch.sum(restframe_vecs[..., PX:]**2, dim=-1)**0.5  # (B, M)
        restframe_vecs[..., E] = torch.maximum(restframe_vecs[..., E], restframe_p + self.eps)

        # create boost objects
        restframe_m = (restframe_vecs[..., E]**2 - restframe_p**2)**0.5  # (B, M)
        gamma = restframe_vecs[..., E] / restframe_m  # (B, M)
        beta = restframe_p / restframe_vecs[..., E]  # (B, M)
        beta_vecs = restframe_vecs[..., PX:] / restframe_vecs[..., E, None]  # (B, M, 3)
        n_vecs = beta_vecs / beta[..., None]  # (B, M, 3)
        e_vecs = torch.cat([torch.ones_like(n_vecs[..., :1]), -n_vecs], dim=-1)  # (B, M, 4)

        # build Lambda
        Lambda = self.I4 + (
            (self.U + gamma[..., None, None]) *
            (self.U1 * beta[..., None, None] - self.U) *
            (e_vecs[..., None] * e_vecs[..., None, :])
        )  # (B, M, 4, 4)

        # apply boosting
        boosted_vecs = (Lambda @ particle_vecs[..., None])[..., 0]

        # hook to update boosted vectors if desired
        boosted_vecs = self.update_boosted_vectors(boosted_vecs)

        # cached feature provision
        cache = {}
        def get(feature: str) -> torch.Tensor:
            # check cache first
            if feature in cache:
                return cache[feature]
            # live feature access
            if feature == "e":
                return boosted_vecs[..., E]
            if feature == "px":
                return boosted_vecs[..., PX]
            if feature == "py":
                return boosted_vecs[..., PY]
            if feature == "pz":
                return boosted_vecs[..., PZ]
            # cached  access
            if feature == "pt2":
                f = get("px")**2 + get("py")**2
            elif feature == "pt":
                f = get("pt2")**0.5
            elif feature == "p2":
                f = get("pt2") + get("pz")**2
            elif feature == "p":
                f = get("p2")**0.5
            elif feature == "eta":
                f = torch.atanh(get("pz") / get("p"))
            elif feature == "phi":
                f = torch.atan2(get("py"), get("px"))
            elif feature == "m":
                f = (torch.maximum(get("e")**2, get("p2")) - get("p"))**0.5
            elif feature == "pair_cos":
                boosted_pvecs = boosted_vecs[..., PX:]  # (B, M, 3)
                boosted_p = get("p")
                f = (
                    (boosted_pvecs @ boosted_pvecs.transpose(1, 2)) /
                    (boosted_p[..., None] @ boosted_p[:, None, :])
                )[..., self.lower_tril]  # (B, (M**2-M)/2)
            elif feature == "pair_dr":
                boosted_phi = get("phi")
                boosted_eta = get("eta")
                boosted_dphi = abs(boosted_phi[..., None] - boosted_phi[:, None, :])  # (B, M, M)
                boosted_dphi = boosted_dphi[..., self.lower_tril]  # (B, (M**2-M)/2)
                boosted_dphi = torch.where(boosted_dphi > torch.pi, 2 * torch.pi - boosted_dphi, boosted_dphi)
                boosted_deta = boosted_eta[..., None] - boosted_eta[:, None, :]  # (B, M, M)
                boosted_deta = boosted_deta[..., self.lower_tril]  # (B, (M**2-M)/2)
                f = (boosted_dphi**2 + boosted_deta**2)**0.5
            else:
                raise RuntimeError(f"unknown feature '{feature}'")
            # cache and return
            cache[feature] = f
            return f

        # when not clipping weights, boosted vectors can have e < p
        if not self.clip_weights:
            boosted_vecs[..., E] = torch.maximum(boosted_vecs[..., E], get("p") + self.eps)

        # collect and combine features
        features = torch.cat([get(feature) for feature in self.features], dim=1)  # (B, F)

        return features


if __name__ == "__main__":
    # hyper-parameters
    N = 10
    M = 5
    bs = 512

    # sample test vectors (simulate numpy's random)
    px = torch.randn(bs, N) * 80.0
    py = torch.randn(bs, N) * 80.0
    pz = torch.randn(bs, N) * 120.0
    m = torch.rand(bs, N) * 50.5 - 0.5
    m = torch.where(m < 0, torch.zeros_like(m), m)
    e = (m**2 + px**2 + py**2 + pz**2)**0.5

    lbn = LBN(N, M, features=LBN.KNOWN_FEATURES)
    feats = lbn(e, px, py, pz)
    print("features shape:", feats.shape)
