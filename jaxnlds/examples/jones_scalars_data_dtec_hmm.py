from jaxnlds.nlds_smoother import NonLinearDynamicsSmoother
from jaxnlds.forward_updates import TecLinearPhaseNestedSampling
import jax.numpy as jnp
from jax import random, jit
from functools import partial
import pylab as plt


def main():

    Gamma0, Omega, Sigma, T, Y_obs, amp, mu0, tec, freqs = generate_data()

    hmm = NonLinearDynamicsSmoother(TecLinearPhaseNestedSampling(freqs))
    hmm = jit(partial(hmm, tol=1., maxiter=2, omega_window=11, sigma_window=5, momentum=0.,
                      omega_diag_range=(0, 20.), sigma_diag_range=(0, jnp.inf)))
    #
    # with disable_jit():
    keys = random.split(random.PRNGKey(0), T)
    res = hmm(Y_obs, Sigma, mu0, Gamma0, Omega, amp, keys)

    print(res.converged, res.niter)
    plt.plot(tec, label='true tec')
    plt.plot(res.post_mu[:, 0], label='infer tec')
    plt.fill_between(jnp.arange(T),
                     res.post_mu[:, 0] - jnp.sqrt(res.post_Gamma[:, 0, 0]),
                     res.post_mu[:, 0] + jnp.sqrt(res.post_Gamma[:, 0, 0]),
                     alpha=0.5)
    plt.legend()
    plt.show()

    plt.plot(jnp.sqrt(res.post_Gamma[:, 0, 0]))
    plt.title("Uncertainty tec")
    plt.show()

    plt.plot(tec - res.post_mu[:, 0], label='infer')
    plt.fill_between(jnp.arange(T),
                     (tec - res.post_mu[:, 0]) - jnp.sqrt(res.post_Gamma[:, 0, 0]),
                     (tec - res.post_mu[:, 0]) + jnp.sqrt(res.post_Gamma[:, 0, 0]),
                     alpha=0.5)
    plt.title("Residual tec")
    plt.legend()
    plt.show()
    plt.plot(jnp.sqrt(res.Omega[:, 0, 0]))
    plt.title("omega")
    plt.show()
    plt.plot(jnp.mean(jnp.sqrt(jnp.diagonal(res.Sigma, axis2=-2, axis1=-1)), axis=-1))
    plt.title("mean sigma")
    plt.show()


def generate_data():
    T = 10
    tec = jnp.cumsum(10. * random.normal(random.PRNGKey(0),shape=(T,)))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = jnp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * random.normal(random.PRNGKey(1), shape=Y.shape)
    # Y_obs[500:550:2, :] += 3. * random.normal(random.PRNGKey(1),shape=Y[500:550:2, :].shape)
    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = jnp.diag(jnp.array([5.]))**2
    mu0 = jnp.zeros(1)
    Gamma0 = jnp.diag(jnp.array([200.]))**2
    amp = jnp.ones_like(phase)
    return Gamma0, Omega, Sigma, T, Y_obs, amp, mu0, tec, freqs


if __name__ == '__main__':
    main()