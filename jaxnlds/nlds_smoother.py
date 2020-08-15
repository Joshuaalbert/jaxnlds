from jaxnlds.forward_updates.forward_update import ForwardUpdateEquation
from jaxnlds.utils import windowed_mean, batched_diag, batched_multi_dot
import jax.numpy as jnp
from jax.lax import scan, while_loop
from jax import vmap, jacobian
from collections import namedtuple
from typing import NamedTuple


class NonLinearDynamicsSmootherResults(NamedTuple):
    converged: bool  # Whether termination due to convergence
    niter: int  # number of iterations used
    post_mu: jnp.ndarray  # posterior mean
    post_Gamma: jnp.ndarray  # posterior convariance
    Sigma: jnp.ndarray  # estimated obs. covariance
    Omega: jnp.ndarray  # estimated transition covariance
    mu0: jnp.ndarray  # estimated initial mu
    Gamma0: jnp.ndarray  # estimate initial covariance


class NonLinearDynamicsSmoother(object):
    def __init__(self, forward_update_equation: ForwardUpdateEquation):
        self.forward_update_equation = forward_update_equation

        def value_and_jacobian(fun):
            jac = jacobian(fun)

            def f(x, *control_params):
                return fun(x, *control_params), jac(x, *control_params)

            return f

        self.batched_value_and_jac = vmap(value_and_jacobian(forward_update_equation.forward_model),
                                          [0] + [0] * forward_update_equation.num_control_params, (0, 0))

    def clip_covariance_diag(self, cov, lo, hi):
        """
        Clips the standard-deviation on the diagonal of cov.
        Args:
            cov: [B, M, M]
            lo: float, standard-dev low value
            hi: float, standard-dev high value

        Returns:
            [B, M, M] covarinace with clipped standard devs.

        """
        variance = batched_diag(cov)
        clipped_variance = jnp.clip(variance, jnp.square(lo), jnp.square(hi))
        add_amount = clipped_variance - variance
        return cov + batched_diag(add_amount)

    def __call__(self, Y, Sigma, mu0, Gamma0, Omega, *control_params, maxiter=None, tol=1e-5, momentum=0.,
                 omega_diag_range=(0, jnp.inf), sigma_diag_range=(0, jnp.inf), omega_window=3, sigma_window=3,
                 beg=None, quack=None):
        """
        Perform HMM inference on a sequence of data.

        Args:
            Y: [T, N] Data of dimension N sequence of length T
            Sigma: Initial estimate of observational covariance broadcastable to [T, N, N].
            mu0: [K] or [T, K] Estimate of initial timestep mean. If shape [T, K] it is taken to be initial estimate of
                post_mu.
            Gamma0: [K, K] or [T, K, K] Estimate of initial timestep state covariance. If shape [T, K, K] it is
                taken to be initial estimate of post_Gamma.
            Omega: Initial estimate of Levy step covariance broadcastable to [T, K, K].
            *control_params: List of control params with first dim size T.
            maxiter: int or None. Number of iterations to run, or until convergence if None. Note, convregence may not
                occur.
            tol: Convergence tolerance in post_mu broadcastable to [K]. Norm_inf is used to check.
            momentum: Momentum broadcastable to [4]. Momentum for [mu0, Gamma0, Omega, Sigma]
            omega_diag_range: 2-tuple of lower and upper bounds for omega, with each element broadcastable to [K].
            sigma_diag_range:2-tuple of lower and upper bounds for sigma, with each element broadcastable to [N].
            omega_window: int or None. Moving average window size for Omega. In None then use whole data sequence.
            sigma_window: int or None. Moving average window size Sigma. In None then use whole data sequence.
            beg: int, prepad `beg` elements to avoid large initial uncertainties in the sequence.
            quack: int, postpad `quack` elements to avoid large final uncertainties in the sequence.

        Returns:
            NonLinearDynamicsSmootherResults containing the results of the run.
                converged: bool  # Whether termination due to convergence
                niter: int  # number of iterations used
                post_mu: jnp.ndarray  # posterior mean
                post_Gamma: jnp.ndarray  # posterior convariance
                Sigma: jnp.ndarray  # estimated obs. covariance
                Omega: jnp.ndarray  # estimated transition covariance
                mu0: jnp.ndarray  # estimated initial mu
                Gamma0: jnp.ndarray  # estimate initial covariance

        """
        NonLinearDynamicsSmootherState = namedtuple('NonLinearDynamicsSmootherState',
                                                    ['done', 'i', 'Sigma_i1', 'post_mu', 'post_Gamma', 'Omega_i1',
                                                     'mu0_i1', 'Gamma0_i1'])
        if maxiter is None:
            maxiter = jnp.inf
        if maxiter <= 0:
            raise ValueError("maxiter {} should be > 0".format(maxiter))
        tol = jnp.array(tol)
        if jnp.any(tol) < 0:
            raise ValueError("tol {} should be > 0".format(tol))
        omega_diag_range = (jnp.array(omega_diag_range[0]), jnp.array(omega_diag_range[1]))
        sigma_diag_range = (jnp.array(sigma_diag_range[0]), jnp.array(sigma_diag_range[1]))

        momentum = jnp.array(momentum)
        momentum = jnp.broadcast_to(momentum, (4,))

        if jnp.any(momentum > 1.) | jnp.any(momentum < 0.):
            raise ValueError("Momentum {} must be in (0,1)".format(momentum))


        Sigma = jnp.broadcast_to(Sigma, Y.shape[0:1] + Sigma.shape[-2:])
        Omega = jnp.broadcast_to(Omega, Y.shape[0:1] + Omega.shape[-2:])
        mu0 = jnp.broadcast_to(mu0, Y.shape[0:1] + Omega.shape[-1:])
        Gamma0 = jnp.broadcast_to(Gamma0, Y.shape[0:1] + Omega.shape[-2:])

        if beg is None:
            beg = 0
        if quack is None:
            quack = 0
        beg = jnp.array(beg)
        quack = jnp.array(quack)

        def _pad(v, beg, quack):
            dims = len(v.shape)
            pad_width = tuple([(beg, quack)] + [(0, 0)] * (dims - 1))
            return jnp.pad(v, pad_width, mode='reflect')

        # technique to avoid large uncertainties at the start and end
        Y = _pad(Y, beg, quack)
        Sigma = _pad(Sigma, beg, quack)
        Omega = _pad(Omega, beg, quack)
        mu0 = _pad(mu0, beg, quack)
        Gamma0 = _pad(Gamma0, beg, quack)
        control_params = [_pad(v, beg, quack) for v in control_params]



        state = NonLinearDynamicsSmootherState(
            done=False,
            i=0,
            post_mu=mu0,
            post_Gamma=Gamma0,
            Sigma_i1=Sigma,
            Omega_i1=Omega,
            mu0_i1=mu0[0,...],
            Gamma0_i1=Gamma0[0,...])

        def body(state):
            prior_Gamma, post_mu_f, post_Gamma_f = self.forward_filter(
                Y, state.Sigma_i1, state.mu0_i1, state.Gamma0_i1,
                state.Omega_i1, *control_params)

            post_mu_b, post_Gamma_b, inter_Gamma = self.backward_filter(prior_Gamma, post_mu_f, post_Gamma_f)

            mu0_i, Gamma0_i, Sigma_i, Omega_i = self.parameter_estimation(
                post_mu_b, post_Gamma_b, inter_Gamma, Y,
                *control_params, omega_window=omega_window,
                sigma_window=sigma_window)

            mu0_i = state.mu0_i1 * momentum[0] + (1. - momentum[0]) * mu0_i
            Gamma0_i = state.Gamma0_i1 * momentum[1] + (1. - momentum[1]) * Gamma0_i
            Omega_i = self.clip_covariance_diag(
                state.Omega_i1 * momentum[2] + (1. - momentum[2]) * Omega_i, omega_diag_range[0], omega_diag_range[1])
            Sigma_i = self.clip_covariance_diag(
                state.Sigma_i1 * momentum[3] + (1. - momentum[3]) * Sigma_i, sigma_diag_range[0], sigma_diag_range[1])

            max_norm = jnp.max(jnp.linalg.norm(state.post_mu - post_mu_b, axis=-1), axis=0)
            converged = jnp.all(max_norm < tol) & (state.i > 0)

            state = state._replace(
                done=converged,
                i=state.i + 1, post_mu=post_mu_b, post_Gamma=post_Gamma_b,
                Sigma_i1=Sigma_i, Omega_i1=Omega_i, mu0_i1=mu0_i, Gamma0_i1=Gamma0_i)

            return state

        state = while_loop(
            lambda state: (~state.done) & (state.i < maxiter),
            body,
            state)

        def _trim(v, beg, quack):
            T = v.shape[0]
            return v[beg:T-quack, ...]

        return NonLinearDynamicsSmootherResults(
            converged=state.done,
            niter=state.i,
            post_mu=_trim(state.post_mu, beg, quack),
            post_Gamma=_trim(state.post_Gamma, beg, quack),
            Sigma=_trim(state.Sigma_i1, beg, quack),
            Omega=_trim(state.Omega_i1, beg, quack),
            mu0=state.post_mu[beg, ...],#actual estimate of beginning is this
            Gamma0=state.post_Gamma[beg, ...]#actual estimate of beginning is this
        )

    def forward_filter(self, Y, Sigma, mu0, Gamma0, Omega, *control_params):
        ForwardFilterState = namedtuple('ForwardFilterState', ['n', 'post_mu_n1', 'post_Gamma_n1'])
        ForwardFilterResult = namedtuple('ForwardFilterResult', ['prior_Gamma', 'post_mu', 'post_Gamma'])
        ForwardFilterX = namedtuple('ForwardFilterX', ['Y', 'Sigma', 'Omega', 'control_params'])

        state = ForwardFilterState(n=0,
                                   post_mu_n1=mu0,
                                   post_Gamma_n1=Gamma0)

        X = ForwardFilterX(Y=Y, Sigma=Sigma, Omega=Omega, control_params=control_params)

        def f(state, X):
            prior_mu_n = state.post_mu_n1
            prior_Gamma_n = state.post_Gamma_n1 + X.Omega

            post_mu_n, post_Gamma_n = self.forward_update_equation.E_update(
                prior_mu_n,
                prior_Gamma_n,
                X.Y,
                X.Sigma,
                *X.control_params)

            state = state._replace(n=state.n + 1,
                                   post_mu_n1=post_mu_n,
                                   post_Gamma_n1=post_Gamma_n)

            result = ForwardFilterResult(prior_Gamma=prior_Gamma_n, post_mu=post_mu_n, post_Gamma=post_Gamma_n)
            return state, result

        final_state, results = scan(f, state, X)

        return results

    def backward_filter(self, prior_Gamma, post_mu_f, post_Gamma_f):
        BackPropState = namedtuple('BackPropState', ['n1', 'post_mu_n', 'post_Gamma_n'])
        BackPropResult = namedtuple('BackPropResult', ['post_mu', 'post_Gamma', 'JT'])
        BackPropX = namedtuple('BackPropX', ['post_mu_f_n1', 'post_Gamma_f_n1', 'prior_Gamma_n'])

        X = BackPropX(post_mu_f_n1=post_mu_f[:-1, ...],
                      post_Gamma_f_n1=post_Gamma_f[:-1, ...],
                      prior_Gamma_n=prior_Gamma[1:, ...])

        state = BackPropState(n1=post_mu_f.shape[0] - 2,
                              post_mu_n=post_mu_f[-1, ...],
                              post_Gamma_n=post_Gamma_f[-1, ...])

        def f_back_prop(state, X):
            JT_t1 = jnp.linalg.solve(X.prior_Gamma_n, X.post_Gamma_f_n1)
            post_mu_n1 = X.post_mu_f_n1 + jnp.dot(JT_t1.T, state.post_mu_n - X.post_mu_f_n1)
            post_Gamma_n1 = X.post_Gamma_f_n1 + jnp.linalg.multi_dot(
                [JT_t1.T, state.post_Gamma_n - X.prior_Gamma_n, JT_t1])

            state = state._replace(n1=state.n1 - 1,
                                   post_mu_n=post_mu_n1,
                                   post_Gamma_n=post_Gamma_n1)

            result = BackPropResult(post_mu=post_mu_n1,
                                    post_Gamma=post_Gamma_n1,
                                    JT=JT_t1)
            return state, result

        final_state, results = scan(f_back_prop, state, X, reverse=True)

        post_mu = jnp.concatenate([results.post_mu, post_mu_f[-1:, ...]], axis=0)
        post_Gamma = jnp.concatenate([results.post_Gamma, post_Gamma_f[-1:, ...]], axis=0)

        InterGammaState = namedtuple('InterGammaState', ['n2', 'inter_Gamma_nn1'])
        InterGammaResult = namedtuple('InterGammaResult', ['inter_Gamma'])
        InterGammaX = namedtuple('InterGammaX', ['JT_n2', 'JT_n1', 'post_Gamma_f_n1'])

        # V_T,T-1^T = V_T^T (V_T^T-1)^{-1} V_T-1^T-1
        inter_Gamma_nn1 = jnp.dot(post_Gamma_f[-1, :, :],
                                  jnp.linalg.solve(prior_Gamma[-1, :, :], post_Gamma_f[-2, :, :]))

        state = InterGammaState(n2=post_mu_f.shape[0] - 3,
                                inter_Gamma_nn1=inter_Gamma_nn1)
        X = InterGammaX(JT_n2=results.JT[:-1, :, :], JT_n1=results.JT[1:, :, :],
                        post_Gamma_f_n1=post_Gamma_f[1:-1, :, :])

        def f_inter_Gamma(state, X):
            a = X.post_Gamma_f_n1 + jnp.dot(X.JT_n1.T, state.inter_Gamma_nn1 - X.post_Gamma_f_n1)
            b = jnp.dot(a, X.JT_n2)
            state = state._replace(n2=state.n2 - 1,
                                   inter_Gamma_nn1=b)
            result = InterGammaResult(inter_Gamma=b)
            return state, result

        final_state, results = scan(f_inter_Gamma, state, X, reverse=True)
        inter_Gamma = jnp.concatenate([results.inter_Gamma, inter_Gamma_nn1[None, :, :]], axis=0)

        BackwardFilterResult = namedtuple('BackwardFilterResult', ['post_mu', 'post_Gamma', 'inter_Gamma'])

        return BackwardFilterResult(post_mu=post_mu, post_Gamma=post_Gamma, inter_Gamma=inter_Gamma)

    def parameter_estimation(self, post_mu_b, post_Gamma_b, inter_Gamma, Y, *control_params,
                             omega_window=1, sigma_window=1):
        """
        Maximise E[log likelihood] over posterior.

        Args:
            post_mu_b:
            post_Gamma_b:
            inter_Gamma:
            Y:
            *control_params:
            omega_window:
            sigma_window:

        Returns:

        """

        MstepResult = namedtuple('MstepResult', ['mu0', 'Gamma0', 'Sigma', 'Omega'])

        mu0 = post_mu_b[0, ...]
        Gamma0 = post_Gamma_b[0, ...]

        dmu = (post_mu_b[1:, :] - post_mu_b[:-1, :])
        Omega_new = dmu[:, :, None] * dmu[:, None, :] + post_Gamma_b[1:, :, :] \
                    + post_Gamma_b[:-1, :, :] - inter_Gamma - jnp.transpose(inter_Gamma, (0, 2, 1))
        Omega_new0 = jnp.mean(Omega_new, axis=0, keepdims=True)
        Omega_new = windowed_mean(Omega_new, omega_window, mode='reflect')
        Omega_new = jnp.concatenate([Omega_new0, Omega_new], axis=0)

        f, D = self.batched_value_and_jac(post_mu_b, *control_params)

        dy = Y - f

        Sigma_new = dy[:, :, None] * dy[:, None, :] + batched_multi_dot([D,
                                                                         post_Gamma_b,
                                                                         jnp.transpose(D, (0, 2, 1))])
        Sigma_new = windowed_mean(Sigma_new, sigma_window, mode='reflect')

        results = MstepResult(mu0=mu0, Gamma0=Gamma0, Sigma=Sigma_new, Omega=Omega_new)

        return results
