from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from src.eigenvalue_decomposition import symmetric_eigh
from src.polar_decomposition import polar_decomposition


@partial(
    jax.jit,
    static_argnames=[
        "polar_method",
        "eps",
    ],
)
def svd_tikhonov(
    A: jnp.array,
    polar_method: str="cans",
    eps: float=1e-8,
) -> List[jnp.array]:
    n_start = A.shape[0]
    if A.shape[0] < A.shape[1]:
        A = A.T
    W, H = polar_decomposition(
        A,
        impl=polar_method,
        eps=eps,
    )

    V, s = symmetric_eigh(H)
    U = W @ V

    U = U[:, jnp.arange(U.shape[-1] - 1, -1, -1)]
    V = V[:, jnp.arange(V.shape[-1] - 1, -1, -1)]
    s = s[jnp.arange(s.shape[-1] - 1, -1, -1)]

    V = jnp.where(s < 0, -V, V)
    s = jnp.abs(s)

    idx = jnp.argsort(s, descending=True)

    U = U[:, idx]
    V = V[:, idx]
    s = s[idx]

    if n_start == A.shape[0]:
        return U, s, V.T
    else:
        return V, s, U.T
