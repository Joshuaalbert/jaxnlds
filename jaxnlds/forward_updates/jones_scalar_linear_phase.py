from jaxnlds.forward_updates.forward_update import ForwardUpdateEquation
from jaxnlds.optimize import minimize
from jaxnlds.utils import deconstrain_std, constrain_std, scalar_KL, value_and_jacobian
from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, MVNPrior, HalfLaplacePrior

from typing import NamedTuple

from jax import numpy as jnp
from jax.lax import while_loop
from jax.scipy.linalg import solve_triangular


class LinearPhaseNestedSampling(ForwardUpdateEquation):
    """
    Uses nested sampling to compute the posterior of a linear phase model for
    Jones scalars, using the Gaussian sample mean and covariance to approximate the solution.

    Specifically, the model is:

        m ~ N[mu, Gamma] # hidden parameter vector and Gamma is the prior (full) covariance
        phase(nu) = m @ f(nu) # linear phase model
        g = amp * exp[1j*phase]
        Y ~ N[{Re(g), Im(g)}, Sigma] # where Sigma is diagonal observational error covariance matrix.
    """

    def __init__(self, freqs, *args, **kwargs):
        self.freqs = freqs

    @property
    def num_control_params(self):
        """
        amp and key
        """
        return 2

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        raise NotImplementedError()

    @property
    def _phase_basis_size(self):
        raise NotImplementedError()

    def forward_model(self, mu, *control_params):
        """
        Return the model data.
        Args:
            mu: [K]

        Returns:
            Model data [N]

        """
        amp = control_params[0]
        f = self._phase_basis(self.freqs)  # Nf,K
        phase = jnp.dot(f, mu)  # Nf
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=0)

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        # amp = control_params[0]
        key = control_params[1]

        prior_chain = PriorChain() \
            .push(MVNPrior('param', prior_mu, prior_Gamma))
            # .push(HalfLaplacePrior('uncert', jnp.sqrt(jnp.mean(jnp.diag(Sigma)))))

        def log_normal(x, mean, cov):


            dx = x - mean
            # L = jnp.linalg.cholesky(cov)
            # dx = solve_triangular(L, dx, lower=True)
            L = jnp.sqrt(jnp.diag(cov))
            dx = dx / L
            return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
                   - 0.5 * dx @ dx

        def log_likelihood(param, **kwargs):
            Y_model = self.forward_model(param, *control_params)
            # Sigma = uncert**2 * jnp.eye(Y.shape[-1])
            return log_normal(Y_model, Y, Sigma)

        ns = NestedSampler(log_likelihood, prior_chain, sampler_name='whitened_ellipsoid')
        results = ns(key, self._phase_basis_size * 15, max_samples=1e5, collect_samples=False,
                     termination_frac=0.01, stoachastic_uncertainty=True)

        post_mu = results.param_mean['param']
        post_Gamma = results.param_covariance['param']

        return post_mu, post_Gamma


class TecLinearPhaseNestedSampling(LinearPhaseNestedSampling):
    def __init__(self, freqs, *args, **kwargs):
        super(TecLinearPhaseNestedSampling, self).__init__(freqs, *args, **kwargs)
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        return self.tec_conv[:, None]

    @property
    def _phase_basis_size(self):
        return 1


class TecConstLinearPhaseNestedSampling(LinearPhaseNestedSampling):
    def __init__(self, freqs, *args, **kwargs):
        super(TecConstLinearPhaseNestedSampling, self).__init__(freqs, *args, **kwargs)
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        return jnp.concatenate([self.tec_conv[:, None], jnp.ones((freqs.shape[0], 1))], axis=1)

    @property
    def _phase_basis_size(self):
        return 2


class TecConstClockLinearPhaseNestedSampling(LinearPhaseNestedSampling):
    def __init__(self, freqs, *args, **kwargs):
        super(TecConstClockLinearPhaseNestedSampling, self).__init__(freqs, *args, **kwargs)
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs
        self.clock_conv = jnp.pi * 2 * freqs

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        return jnp.concatenate([self.tec_conv[:, None], jnp.ones((freqs.shape[0], 1)), self.clock_conv[:, None]],
                               axis=1)

    @property
    def _phase_basis_size(self):
        return 3


###
# Variational linear phase model

class LinearPhase(ForwardUpdateEquation):
    """
    Uses variational inference to compute a Gaussian approximation to the posterior of a linear phase model for
    Jones scalars. Uses BFGS optimiser to solve for maximum likelihood parameters of the variational posterior.
    Suffers from many local minima because of phase wrapping.

    Specifically, the model is:

        m ~ N[mu, Gamma] # hidden parameter vector and Gamma is diagonal in this version
        phase(nu) = m @ f(nu) # linear phase model
        g = amp * exp[1j*phase]
        Y ~ N[{Re(g), Im(g), Sigma] # where Sigma is diagonal observational error covariance matrix.
    """

    def __init__(self, freqs, freeze_params=()):
        self.freqs = freqs
        if isinstance(freeze_params, int):
            freeze_params = (freeze_params,)
        self.freeze_params = tuple(freeze_params)

    @property
    def num_control_params(self):
        return 1

    def initial_parameters(self, prior_mu, prior_gamma):
        prior_gamma = deconstrain_std(prior_gamma)
        if len(self.freeze_params) > 0:
            mu0 = [prior_mu[i:i + 1] for i in range(self._phase_basis_size) if i not in self.freeze_params]
            gamma0 = [prior_gamma[i:i + 1] for i in range(self._phase_basis_size) if i not in self.freeze_params]
            x0 = jnp.concatenate(mu0 + gamma0)
            frozen_mu = [prior_mu[i:i + 1] for i in range(self._phase_basis_size) if i in self.freeze_params]
            frozen_gamma = [prior_gamma[i:i + 1] for i in range(self._phase_basis_size) if i in self.freeze_params]
            frozen_params = jnp.concatenate(frozen_mu + frozen_gamma)
        else:
            x0 = jnp.concatenate([prior_mu, prior_gamma])
            frozen_params = None
        return x0, frozen_params

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        raise NotImplementedError()

    @property
    def _phase_basis_size(self):
        raise NotImplementedError()

    def parse_params(self, params, frozen_params):
        if frozen_params is not None:
            num_params = self._phase_basis_size - len(self.freeze_params)
            num_frozen = len(self.freeze_params)
            mu = params[:num_params]
            gamma = constrain_std(params[num_params:])
            frozen_mu = frozen_params[:num_frozen]
            frozen_gamma = constrain_std(frozen_params[num_frozen:])
            joint_mu = []
            joint_gamma = []
            frozen_count = 0
            param_count = 0
            for i in range(self._phase_basis_size):
                if i in self.freeze_params:
                    joint_mu.append(frozen_mu[frozen_count:frozen_count + 1])
                    joint_gamma.append(frozen_gamma[frozen_count:frozen_count + 1])
                    frozen_count += 1
                else:
                    joint_mu.append(mu[param_count:param_count + 1])
                    joint_gamma.append(gamma[param_count:param_count + 1])
                    param_count += 1
            joint_mu = jnp.concatenate(joint_mu)
            joint_gamma = jnp.concatenate(joint_gamma)
            return joint_mu, joint_gamma
        mu = params[:self._phase_basis_size]
        gamma = constrain_std(params[self._phase_basis_size:])
        return mu, gamma

    def forward_model(self, mu, *control_params):
        """
        Return the model data.
        Args:
            mu: [K]

        Returns:
            Model data [N]

        """
        amp = control_params[0]
        f = self._phase_basis(self.freqs)  # Nf,K
        phase = jnp.dot(f, mu)  # Nf
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=0)

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        amp = control_params[0]

        sigma = jnp.sqrt(jnp.diag(Sigma))
        prior_gamma = jnp.sqrt(jnp.diag(prior_Gamma))
        x0, frozen_params = self.initial_parameters(prior_mu, prior_gamma)

        def neg_elbo(params, frozen_params):
            mu, gamma = self.parse_params(params, frozen_params)
            return self.neg_elbo(self.freqs, Y, sigma, amp, mu, gamma, prior_mu, prior_gamma)

        def do_minimisation(x0):
            result = minimize(neg_elbo, x0, args=(frozen_params,), method='BFGS',
                              options=dict(ls_maxiter=100, gtol=1e-12))
            return result.x

        x1 = do_minimisation(x0)

        post_mu, post_gamma = self.parse_params(x1, frozen_params)
        post_Gamma = jnp.diag(jnp.square(post_gamma))

        return post_mu, post_Gamma

    def neg_elbo(self, freqs, Y_obs, sigma, amp, mu, gamma, mu_prior, gamma_prior):
        return scalar_KL(mu, gamma, mu_prior, gamma_prior) - self.var_exp(freqs, Y_obs, sigma, amp, mu, gamma)

    def var_exp(self, freqs, Y_obs, sigma, amp, mu, gamma):
        """
        Computes variational expectation
        Args:
            freqs: [Nf]
            Y_obs: [Nf]
            sigma: [Nf]
            amp: [Nf]
            mu: [M]
            gamma: [M]

        Returns: scalar

        """
        f = self._phase_basis(self.freqs)  # Nf,M
        Nf = freqs.size
        Sigma_real = jnp.square(sigma[:Nf])
        Sigma_imag = jnp.square(sigma[Nf:])
        Yreal = Y_obs[:Nf]
        Yimag = Y_obs[Nf:]
        a = jnp.reciprocal(Sigma_real)
        b = jnp.reciprocal(Sigma_imag)
        constant = -Nf * jnp.log(2. * jnp.pi)
        logdet = -jnp.sum(jnp.log(sigma))

        phi = jnp.dot(f, mu)
        theta = jnp.dot(jnp.square(f), jnp.square(gamma))

        exp_cos = jnp.exp(-0.5 * theta) * jnp.cos(phi)
        exp_sin = jnp.exp(-0.5 * theta) * jnp.sin(phi)
        exp_cos2 = 0.5 * (jnp.exp(-2. * theta) * jnp.cos(2. * phi) + 1.)

        negtwo_maha = a * (jnp.square(Yreal) - 2. * amp * Yreal * exp_cos) \
                      + b * (jnp.square(Yimag) - 2. * amp * Yimag * exp_sin) \
                      + ((a - b) * exp_cos2 + b) * jnp.square(amp)

        return constant + logdet - 0.5 * jnp.sum(negtwo_maha)


class TecLinearPhase(LinearPhase):
    def __init__(self, freqs, *args, **kwargs):
        super(TecLinearPhase, self).__init__(freqs, *args, **kwargs)
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        return self.tec_conv[:, None]

    @property
    def _phase_basis_size(self):
        return 1


class TecConstLinearPhase(LinearPhase):
    def __init__(self, freqs, *args, **kwargs):
        super(TecConstLinearPhase, self).__init__(freqs, *args, **kwargs)
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        return jnp.concatenate([self.tec_conv[:, None], jnp.ones((freqs.shape[0], 1))], axis=1)

    @property
    def _phase_basis_size(self):
        return 2


###
# Special case of tec only (prefer LinearPhase model instead)

class TecOnlyOriginal(ForwardUpdateEquation):
    """
    Special case of tec only model using variational inference to compute a Gaussian approximation to the posterior.
    Uses BFGS optimiser to solve for maximum likelihood parameters of the variational posterior. Suffers from
    many local minima because of phase wrapping.

    Specifically, the model is:

        tec ~ N[mu, Gamma]
        phase = tec * TEC_CONV / freqs
        g = amp * exp[1j*phase]
        Y ~ N[{Re(g), Im(g), Sigma]
        where Sigma is diagonal observational error covariance matrix.

        Prefer LinearPhase instead.
    """

    def __init__(self, freqs):
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    @property
    def num_control_params(self):
        return 1

    def forward_model(self, mu, *control_params):
        """
        Return the model data.
        Args:
            mu: [K]

        Returns:
            Model data [N]

        """
        amp = control_params[0]
        phase = mu[0] * self.tec_conv
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=0)

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        amp = control_params[0]

        sigma = jnp.sqrt(jnp.diag(Sigma))
        prior_gamma = jnp.sqrt(jnp.diag(prior_Gamma))

        def neg_elbo(params):
            mu, gamma = params
            gamma = constrain_std(gamma)
            res = self.neg_elbo(self.freqs, Y, sigma, amp, mu, gamma, prior_mu, prior_gamma)
            return res

        def do_minimisation(x0):
            result = minimize(neg_elbo, x0, method='BFGS',
                              options=dict(ls_maxiter=100, gtol=1e-6))
            return result.x

        x0 = jnp.concatenate([prior_mu, deconstrain_std(jnp.array([5.]))])
        x1 = do_minimisation(x0)
        # basin = jnp.mean(jnp.abs(jnp.pi / self.tec_conv)) * 0.5
        # num_basin = int(self.tec_scale / basin) + 1
        #
        # obj_try = jnp.stack(
        #     [neg_elbo(jnp.array([x1[0] + i * basin, x1[1]])) for i in range(-num_basin, num_basin + 1, 1)],
        #     axis=0)
        # which_basin = jnp.argmin(obj_try, axis=0)
        # x0_next = jnp.array([x1[0] + (which_basin - float(num_basin)) * basin, x1[1]])
        # x2 = do_minimisation(x0_next)
        x2 = x1

        tec_mean = x2[0]
        tec_uncert = constrain_std(x2[1])

        post_mu = jnp.array([tec_mean])
        post_cov = jnp.array([[tec_uncert ** 2]])

        return post_mu, post_cov

    def neg_elbo(self, freqs, Y_obs, sigma, amp, mu, gamma, mu_prior, gamma_prior):
        return scalar_KL(mu, gamma, mu_prior, gamma_prior) - self.var_exp(freqs, Y_obs, sigma, amp, mu, gamma)

    def var_exp(self, freqs, Y_obs, sigma, amp, mu, gamma):
        """
        Computes variational expectation
        Args:
            freqs: [Nf]
            Y_obs: [Nf]
            sigma: [Nf]
            amp: [Nf]
            mu: scalar
            gamma: scalar

        Returns: scalar

        """
        Nf = freqs.size
        sigma_real = sigma[:Nf]
        sigma_imag = sigma[Nf:]
        m = mu
        l = gamma
        amps = amp
        Yreal = Y_obs[:Nf]
        Yimag = Y_obs[Nf:]
        a = 1. / sigma_real
        b = 1. / sigma_imag
        phi = self.tec_conv * m
        theta = self.tec_conv ** 2 * l * l
        res = -b ** 2 * (amps ** 2 + 2. * Yimag ** 2)
        res += -a ** 2 * (amps ** 2 + 2. * Yreal ** 2)
        res += -4. * jnp.log(2. * jnp.pi / (a * b))
        res += amps * jnp.exp(-2. * theta) * (
                amps * (b ** 2 - a ** 2) * jnp.cos(2. * phi) + 4. * jnp.exp(1.5 * theta) * (
                a ** 2 * Yreal * jnp.cos(phi) + b ** 2 * Yimag * jnp.sin(phi)))
        res *= 0.25
        return jnp.sum(res, axis=-1)


###
# Linearised models use an iterative linearisation of the non-lineararity and then apply analytic Gaussian updates.
# (prefer LinearPhase model instead)
class LinearPhaseLinearised(ForwardUpdateEquation):
    """
    Linearises the forward equation, and, assuming Gaussian errors, iteratively updates the parameters corresponding
    to a linear phase model. Prefer LinearPhase model instead.
    """

    def __init__(self, freqs, tol=1e-4, maxiter=20, momentum=0.):
        self.freqs = freqs
        self.tol = jnp.array(tol)
        self.momentum = jnp.array(momentum)
        self.maxiter = maxiter

    @property
    def num_control_params(self):
        return 1

    def initial_parameters(self, mu0, Gamma0):
        return mu0, Gamma0

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        raise NotImplementedError()

    @property
    def _phase_basis_size(self):
        raise NotImplementedError()

    def forward_model(self, mu, *control_params):
        """
        Return the model data.
        Args:
            mu: [K]

        Returns:
            Model data [N]

        """
        amp = control_params[0]
        f = self._phase_basis(self.freqs)  # Nf,M
        phase = jnp.dot(f, mu)  # Nf
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=0)

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        amp = control_params[0]
        f_and_J = value_and_jacobian(self.forward_model)

        class State(NamedTuple):
            iter: int
            converged: bool
            mu: jnp.ndarray
            Gamma: jnp.ndarray

        state = State(0, False, *self.initial_parameters(prior_mu, prior_Gamma))

        def body(state: State):
            f_0, D = f_and_J(state.mu, amp)
            dY = Y - f_0
            DG = jnp.dot(D, prior_Gamma)
            S = jnp.dot(DG, D.T) + Sigma
            KT = jnp.linalg.solve(S, DG)
            post_mu = prior_mu + jnp.dot(KT.T, dY)
            post_Gamma = prior_Gamma - jnp.linalg.multi_dot([KT.T, D, prior_Gamma])
            converged = jnp.all(jnp.linalg.norm(state.mu - post_mu) < self.tol)
            state = state._replace(converged=converged, iter=state.iter + 1,
                                   mu=state.mu * self.momentum + (1. - self.momentum) * post_mu,
                                   Gamma=state.Gamma * self.momentum + (1. - self.momentum) * post_Gamma)
            return state

        _, _, post_mu, post_Gamma = while_loop(lambda state: (~state.converged) & (state.iter < self.maxiter),
                                               body,
                                               state)

        return post_mu, post_Gamma


class TecLinearPhaseLinearised(LinearPhaseLinearised):
    """
    Linearised model of tec.
    """

    def __init__(self, freqs, *args, **kwargs):
        super(TecLinearPhaseLinearised, self).__init__(freqs, *args, **kwargs)
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    def initial_parameters(self, mu0, Gamma0):
        return mu0, jnp.diag(jnp.array([5.])) ** 2

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        return self.tec_conv[:, None]

    @property
    def _phase_basis_size(self):
        return 1


class TecConstLinearPhaseLinearised(LinearPhaseLinearised):
    """
    Linearised model of tec and constant scalar phase.
    """

    def __init__(self, freqs, *args, **kwargs):
        super(TecConstLinearPhaseLinearised, self).__init__(freqs, *args, **kwargs)
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        return jnp.concatenate([self.tec_conv[:, None], jnp.ones((freqs.shape[0], 1))], axis=1)

    @property
    def _phase_basis_size(self):
        return 2
