from jax import numpy as jnp, numpy as np
from jax import vmap, jacobian
from jax.scipy.linalg import solve_triangular
from jax.scipy.signal import _convolve_nd


def windowed_mean(a, w, mode='reflect'):
    if w is None:
        T = a.shape[0]
        return jnp.broadcast_to(jnp.mean(a, axis=0, keepdims=True), a.shape)
    dims = len(a.shape)
    a = a
    kernel = jnp.reshape(jnp.ones(w)/w, [w]+[1]*(dims-1))
    _w1 = (w-1)//2
    _w2 = _w1 if (w%2 == 1) else _w1 + 1
    pad_width = [(_w1, _w2)] + [(0,0)]*(dims-1)
    a = jnp.pad(a, pad_width=pad_width, mode=mode)
    return _convolve_nd(a,kernel, mode='valid', precision=None)


batched_diag = vmap(jnp.diag, 0, 0)

batched_multi_dot = vmap(jnp.linalg.multi_dot, 0, 0)


def constrain(v, a, b):
    return a + (jnp.tanh(v) + 1) * (b - a) / 2.

def constrain_std(v, vmin = 1e-3):
    return jnp.abs(v) + vmin

def deconstrain_std(v, vmin = 1e-3):
    return jnp.maximum(v - vmin, 0.)


def deconstrain(v, a, b):
    return jnp.arctanh(jnp.clip((v - a) * 2. / (b - a) - 1., -0.999, 0.999))

def constrain_tec(v, vmin = 0.5):
    return jnp.abs(v) + vmin

def deconstrain_tec(v, vmin = 0.5):
    return v - vmin

def constrain_omega(v, lower=0.5, scale=10.):
    return scale * jnp.log(jnp.exp(v) + 1.) + lower

def deconstrain_omega(v, lower=0.5, scale=10.):
    y = jnp.maximum(jnp.exp((v - lower) / scale) - 1., 0.)
    return jnp.maximum(-1e3, jnp.log(y))

def constrain_sigma(v, lower=0.01, scale=0.5):
    return scale * jnp.log(jnp.exp(v) + 1.) + lower

def deconstrain_sigma(v, lower=0.01, scale=0.5):
    y = jnp.maximum(jnp.exp((v - lower) / scale) - 1., 0.)
    return jnp.maximum(-1e3, jnp.log(y))


def scalar_KL(mean, uncert, mean_prior, uncert_prior):
    """
    mean, uncert : [M]
    mean_prior,uncert_prior: [M]
    :return: scalar
    """
    # Get KL
    q_var = np.square(uncert)
    var_prior = np.square(uncert_prior)
    trace = q_var / var_prior
    mahalanobis = np.square(mean - mean_prior) / var_prior
    constant = -1.
    logdet_qcov = np.log(var_prior / q_var)
    twoKL = mahalanobis + constant + logdet_qcov + trace
    prior_KL = 0.5 * twoKL
    return np.sum(prior_KL)


def mvn_kl(mu_a, L_a, mu_b, L_b):
    def squared_frobenius_norm(x):
        return np.sum(np.square(x))

    b_inv_a = solve_triangular(L_b, L_a, lower=True)
    kl_div = (
            np.sum(np.log(np.diag(L_b))) - np.sum(np.log(np.diag(L_a))) +
            0.5 * (-L_a.shape[-1] +
                   squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(
                solve_triangular(L_b, mu_b[:, None] - mu_a[:, None], lower=True))))
    return kl_div


def fill_triangular(x, upper=False):
    m = x.shape[-1]
    if len(x.shape) != 1:
        raise ValueError("Only handles 1D to 2D transformation, because tril/u")
    m = np.int32(m)
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
        raise ValueError('Input right-most shape ({}) does not '
                         'correspond to a triangular matrix.'.format(m))
    n = np.int32(n)
    final_shape = list(x.shape[:-1]) + [n, n]
    if upper:
        x_list = [x, np.flip(x[..., n:], -1)]

    else:
        x_list = [x[..., n:], np.flip(x, -1)]
    x = np.reshape(np.concatenate(x_list, axis=-1), final_shape)
    if upper:
        x = np.triu(x)
    else:
        x = np.tril(x)
    return x


def fill_triangular_inverse(x, upper=False):
    n = x.shape[-1]
    n = np.int32(n)
    m = np.int32((n * (n + 1)) // 2)
    final_shape = list(x.shape[:-2]) + [m]
    if upper:
        initial_elements = x[..., 0, :]
        triangular_portion = x[..., 1:, :]
    else:
        initial_elements = np.flip(x[..., -1, :], axis=-2)
        triangular_portion = x[..., :-1, :]
    rotated_triangular_portion = np.flip(
        np.flip(triangular_portion, axis=-1), axis=-2)
    consolidated_matrix = triangular_portion + rotated_triangular_portion
    end_sequence = np.reshape(
        consolidated_matrix,
        list(x.shape[:-2]) + [n * (n - 1)])
    y = np.concatenate([initial_elements, end_sequence[..., :m - n]], axis=-1)
    return y

def polyfit(x, y, deg):
    """
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.
    """
    order = int(deg) + 1
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")
    rcond = len(x) * jnp.finfo(x.dtype).eps
    lhs = jnp.stack([x ** (deg - i) for i in range(order)], axis=1)
    rhs = y
    scale = jnp.sqrt(jnp.sum(lhs * lhs, axis=0))
    lhs /= scale
    c, resids, rank, s = jnp.linalg.lstsq(lhs, rhs, rcond)
    c = (c.T / scale).T  # broadcast scale coefficients
    return c


def value_and_jacobian(fun):
    jac = jacobian(fun)

    def f(x, *control_params):
        return fun(x, *control_params), jac(x, *control_params)

    return f