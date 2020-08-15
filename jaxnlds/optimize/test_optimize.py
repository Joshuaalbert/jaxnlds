"""
Tests for optimize routines
"""

from numpy.testing import assert_, assert_array_almost_equal, assert_array_almost_equal_nulp
import jax.numpy as jnp
from .line_search import line_search
from .bfgs_minimize import fmin_bfgs
from jax import jit
from scipy.optimize import minimize as smin
import numpy as onp


def assert_wolfe(s, phi, derphi, c1=1e-4, c2=0.9, err_msg=""):
    """
    Check that strong Wolfe conditions apply
    """
    phi1 = phi(s)
    phi0 = phi(0)
    derphi0 = derphi(0)
    derphi1 = derphi(s)
    msg = "s = %s; phi(0) = %s; phi(s) = %s; phi'(0) = %s; phi'(s) = %s; %s" % (
        s, phi0, phi1, derphi0, derphi1, err_msg)

    assert_(phi1 <= phi0 + c1 * s * derphi0, "Wolfe 1 failed: " + msg)
    assert_(abs(derphi1) <= abs(c2 * derphi0), "Wolfe 2 failed: " + msg)


def assert_line_wolfe(x, p, s, f, fprime, **kw):
    assert_wolfe(s, phi=lambda sp: f(x + p * sp),
                 derphi=lambda sp: jnp.dot(fprime(x + p * sp), p), **kw)


def assert_fp_equal(x, y, err_msg="", nulp=50):
    """Assert two arrays are equal, up to some floating-point rounding error"""
    try:
        assert_array_almost_equal_nulp(x, y, nulp)
    except AssertionError as e:
        raise AssertionError("%s\n%s" % (e, err_msg))


def value_and_grad(f, fprime):
    def func(x):
        return f(x), fprime(x)

    return func


class TestLineSearch(object):
    # -- scalar functions; must have dphi(0.) < 0
    def _scalar_func_1(self, s):
        self.fcount += 1
        p = -s - s ** 3 + s ** 4
        dp = -1 - 3 * s ** 2 + 4 * s ** 3
        return p, dp

    def _scalar_func_2(self, s):
        self.fcount += 1
        p = jnp.exp(-4 * s) + s ** 2
        dp = -4 * jnp.exp(-4 * s) + 2 * s
        return p, dp

    def _scalar_func_3(self, s):
        self.fcount += 1
        p = -jnp.sin(10 * s)
        dp = -10 * jnp.cos(10 * s)
        return p, dp

    # -- num_parent-d functions

    def _line_func_1(self, x):
        self.fcount += 1
        f = jnp.dot(x, x)
        df = 2 * x
        return f, df

    def _line_func_2(self, x):
        self.fcount += 1
        f = jnp.dot(x, jnp.dot(self.A, x)) + 1
        df = jnp.dot(self.A + self.A.T, x)
        return f, df

    # --

    def setup_method(self):
        self.scalar_funcs = []
        self.line_funcs = []
        self.N = 20
        self.fcount = 0

        def bind_index(func, idx):
            # Remember Python's closure semantics!
            return lambda *a, **kw: func(*a, **kw)[idx]

        for name in sorted(dir(self)):
            if name.startswith('_scalar_func_'):
                value = getattr(self, name)
                self.scalar_funcs.append(
                    (name, bind_index(value, 0), bind_index(value, 1)))
            elif name.startswith('_line_func_'):
                value = getattr(self, name)
                self.line_funcs.append(
                    (name, bind_index(value, 0), bind_index(value, 1)))

        onp.random.seed(1234)
        self.A = onp.random.randn(self.N, self.N)

    def scalar_iter(self):
        for name, phi, derphi in self.scalar_funcs:
            for old_phi0 in onp.random.randn(3):
                yield name, phi, derphi, old_phi0

    def line_iter(self):
        for name, f, fprime in self.line_funcs:
            k = 0
            while k < 9:
                x = onp.random.randn(self.N)
                p = onp.random.randn(self.N)
                if jnp.dot(p, fprime(x)) >= 0:
                    # always pick a descent direction
                    continue
                k += 1
                old_fv = float(onp.random.randn())
                yield name, f, fprime, x, p, old_fv

    # -- Generic scalar searches

    def test_scalar_search_wolfe2(self):
        for name, phi, derphi, old_phi0 in self.scalar_iter():
            res = line_search(value_and_grad(phi, derphi), 0., 1.)
            s, phi1, derphi1 = res.a_k, res.f_k, res.g_k
            # s, phi1, phi0, derphi1 = ls.scalar_search_wolfe2(
            #     phi, derphi, phi(0), old_phi0, derphi(0))
            assert_fp_equal(phi1, phi(s), name)
            if derphi1 is not None:
                assert_fp_equal(derphi1, derphi(s), name)
            assert_wolfe(s, phi, derphi, err_msg="%s %g" % (name, old_phi0))

    # -- Generic line searches

    def test_line_search_wolfe2(self):
        c = 0
        smax = 512
        for name, f, fprime, x, p, old_f in self.line_iter():
            f0 = f(x)
            g0 = fprime(x)
            self.fcount = 0
            res = line_search(value_and_grad(f, fprime), x, p, old_fval=f0, gfk=g0)
            s = res.a_k
            fc = res.nfev
            gc = res.ngev
            fv = res.f_k
            gv = res.g_k
            # s, fc, gc, fv, ofv, gv = ls.line_search_wolfe2(f, fprime, x, p,
            #                                                g0, f0, old_f,
            #                                                amax=smax)
            # assert_equal(self.fcount, fc+gc)
            assert_array_almost_equal(fv, f(x + s * p), decimal=5)
            if gv is not None:
                assert_array_almost_equal(gv, fprime(x + s * p), decimal=5)
            if s < smax:
                c += 1
        assert_(c > 3)  # check that the iterator really works...

    def test_line_search_wolfe2_bounds(self):
        # See gh-7475

        # For this f and p, starting at a point on axis 0, the strong Wolfe
        # condition 2 is met if and only if the step length s satisfies
        # |x + s| <= c2 * |x|
        f = lambda x: jnp.dot(x, x)
        fp = lambda x: 2 * x
        p = jnp.array([1, 0])

        # Smallest s satisfying strong Wolfe conditions for these arguments is 30
        x = -60 * p
        c2 = 0.5

        res = line_search(value_and_grad(f, fp), x, p, c2=c2)
        s = res.a_k
        # s, _, _, _, _, _ = ls.line_search_wolfe2(f, fp, x, p, amax=30, c2=c2)
        assert_line_wolfe(x, p, s, f, fp)
        assert s >= 30.

        res = line_search(value_and_grad(f, fp), x, p, c2=c2, maxiter=5)
        assert res.failed
        # s=30 will only be tried on the 6th iteration, so this won't converge

    def test_line_search(self):
        import jax

        import jax.numpy as np

        def f(x):
            return np.cos(np.sum(np.exp(-x)) ** 2)

        # assert not line_search(jax.value_and_grad(f), num_per_cluster.ones(2), num_per_cluster.array([-0.5, -0.25])).failed
        xk = np.ones(2)
        pk = np.array([-0.5, -0.25])
        res = line_search(jax.value_and_grad(f), xk, pk, maxiter=100)

        from scipy.optimize.linesearch import line_search_wolfe2

        scipy_res = line_search_wolfe2(f, jax.grad(f), xk, pk)

        # print(scipy_res[0], res.a_k)
        # print(scipy_res[3], res.f_k)

        assert np.isclose(scipy_res[0], res.a_k)
        assert np.isclose(scipy_res[3], res.f_k)

# -- More specific tests

def rosenbrock(np):
    def func(x):
        return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)

    return func


def himmelblau(np):
    def func(p):
        x, y = p
        return (x ** 2 + y - 11.) ** 2 + (x + y ** 2 - 7.) ** 2

    return func


def matyas(np):
    def func(p):
        x, y = p
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

    return func


def eggholder(np):
    def func(p):
        x, y = p
        return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2. + y + 47.))) - x * np.sin(
            np.sqrt(np.abs(x - (y + 47.))))

    return func


class TestBFGS(object):
    # def __init__(self):
    #     pass

    def test_minimize(self):
        # Note, cannot compare step for step with scipy BFGS because our line search is _slightly_ different.

        for maxiter in [None]:
            for func_and_init in [(rosenbrock, jnp.zeros(2)),
                                  (himmelblau, jnp.zeros(2)),
                                  (matyas, jnp.ones(2) * 6.),
                                  (eggholder, jnp.ones(2) * 100.)]:
                func, x0 = func_and_init

                def compare(func, x0):
                    @jit
                    def min_op(x0):
                        result = fmin_bfgs(func(jnp), x0,
                                           options=dict(ls_maxiter=100, maxiter=maxiter, gtol=1e-6))
                        return result

                    jax_res = min_op(x0)

                    scipy_res = smin(func(onp), x0, method='BFGS')

                    assert onp.isclose(scipy_res.x, jax_res.x_k, atol=2e-5).all()

                compare(func, x0)


if __name__ == '__main__':
    TestBFGS().test_minimize()
