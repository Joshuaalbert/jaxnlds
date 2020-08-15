"""The Broyden-Fletcher-Goldfarb-Shanno minimization algorithm.
https://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf
"""

import jax
import jax.numpy as jnp
from jax.lax import while_loop
from .line_search import line_search
from .bfgs_minimize import BFGSResults
from typing import NamedTuple, Optional, Tuple


def fmin_b_bfgs(func, x0, args=(), options=None):
    """
    The BFGS algorithm from
        Algorithm 6.1 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 136-143

    with bounded parameters, using the active set approach from,
        Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995).
        'A Limited Memory Algorithm for Bound Constrained Optimization.'
         SIAM Journal on Scientific Computing, 16(5), 1190–1208.
         doi:10.1137/0916069

        Notes:
            We utilise boolean arithmetic to avoid jax.cond calls which don't work on accelerators.
            A side effect is that we perform more gradient evaluations than scipy's BFGS
        func: callable
            Function of the form f(x) where x is a flat ndarray and returns a real scalar. The function should be
            composed of operations with vjp defined. If func is jittable then fmin_bfgs is jittable. If func is
            not jittable, then _nojit should be set to True.

        x0: ndarray
            initial variable
        args: tuple, optional
            Extra arguments to pass to func as func(x,*args)
        options: Optional dict of parameters
            maxiter: int
                Maximum number of evaluations
            norm: float
                Order of norm for convergence check. Default inf.
            gtol: flat
                Terminates minimization when |grad|_norm < g_tol
            ls_maxiter: int
                Maximum number of linesearch iterations
            bounds: 2-tuple of two vectors specifying the lower and upper bounds.
                e.g. (l, u) where l and u have the same size as x0. For parameters x_i without constraints the
                corresponding l_i=-jnp.inf and u_i=jnp.inf. Specifying l=None or u=None means no constraints on that
                side.

    Returns: BFGSResults

    """

    if options is None:
        options = dict()
    maxiter: Optional[int] = options.get('maxiter', None)
    norm: float = options.get('norm', jnp.inf)
    gtol: float = options.get('gtol', 1e-5)
    ls_maxiter: int = options.get('ls_maxiter', 10)
    bounds: Tuple[jnp.ndarray, jnp.ndarray] = tuple(options.get('bounds', (None, None)))

    state = BFGSResults(converged=False,
                        failed=False,
                        k=0,
                        nfev=0,
                        ngev=0,
                        nhev=0,
                        x_k=x0,
                        f_k=None,
                        g_k=None,
                        H_k=None,
                        status=None,
                        ls_status=jnp.array(0))

    if maxiter is None:
        maxiter = jnp.size(x0) * 200

    d = x0.shape[0]

    l = bounds[0]
    u = bounds[1]
    if l is None:
        l = -jnp.inf * jnp.ones_like(x0)
    if u is None:
        u = jnp.inf * jnp.ones_like(x0)
    l,u = jnp.where(l<u, l, u), jnp.where(l<u,u, l)

    def project(x,l,u):
        return jnp.clip(x,l, u)

    def get_active_set(x, l, u):
        return jnp.where((x==l) | (x==u))

    def func_with_args(x):
        return func(x, *args)

    def get_generalised_Cauchy_point(xk, gk, l, u):
        def func(t):
            return func_with_args(project(xk - t* gk, l, u))

    initial_H = jnp.eye(d)
    initial_H = options.get('hess_inv', initial_H)



    value_and_grad = jax.value_and_grad(func_with_args)

    f_0, g_0 = value_and_grad(x0)
    state = state._replace(f_k=f_0, g_k=g_0, H_k=initial_H, nfev=state.nfev + 1, ngev=state.ngev + 1,
                           converged=jnp.linalg.norm(g_0, ord=norm) < gtol)

    def body(state):
        p_k = -(state.H_k @ state.g_k)
        line_search_results = line_search(value_and_grad, state.x_k, p_k, old_fval=state.f_k, gfk=state.g_k,
                                          maxiter=ls_maxiter)
        state = state._replace(nfev=state.nfev + line_search_results.nfev,
                               ngev=state.ngev + line_search_results.ngev,
                               failed=line_search_results.failed,
                               ls_status=line_search_results.status)
        s_k = line_search_results.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = line_search_results.f_k
        g_kp1 = line_search_results.g_k
        # print(g_kp1)
        y_k = g_kp1 - state.g_k
        rho_k = jnp.reciprocal(y_k @ s_k)

        sy_k = s_k[:, None] * y_k[None, :]
        w = jnp.eye(d) - rho_k * sy_k
        H_kp1 = jnp.where(jnp.isfinite(rho_k),
                          jnp.linalg.multi_dot([w, state.H_k, w.T]) + rho_k * s_k[:, None] * s_k[None, :], state.H_k)

        converged = jnp.linalg.norm(g_kp1, ord=norm) < gtol

        state = state._replace(converged=converged,
                               k=state.k + 1,
                               x_k=x_kp1,
                               f_k=f_kp1,
                               g_k=g_kp1,
                               H_k=H_kp1
                               )

        return state

    state = while_loop(
        lambda state: (~ state.converged) & (~state.failed) & (state.k < maxiter),
        body,
        state)

    state = state._replace(status=jnp.where(state.converged, jnp.array(0),  # converged
                                            jnp.where(state.k == maxiter, jnp.array(1),  # max iters reached
                                                      jnp.where(state.failed, jnp.array(2) + state.ls_status,
                                                                # ls failed (+ reason)
                                                                jnp.array(-1)))))  # undefined

    return state
