"""Stochastic consumption-savings model of Deaton (1991) and three solvers.

The code of Class 16 (``deaton.ipynb``, https://dse.iskh.me/cons-practice) collected
into a module, so that later classes can import it:

    from deaton import deaton, solve_vfi, bellman_discretized, bellman_continuous, \
                       solve_timeiter, euler_residual, accuracy, simulate, interp
    m = deaton()                                  # beta=0.9, R=1.05, sigma=0.5
    c = solve_timeiter(m)                         # policy function on m.grid
    print(accuracy(m, c))                         # mean relative Euler residual

The expectation over the log-normal income is computed by quadrature, chosen by the
argument ``quad`` of the model object: ``quad='hermite'`` (default) for Gauss-Hermite,
exact for the normal distribution of log income, or ``quad='legendre'`` for
Gauss-Legendre on the CDF of income, the rule of Class 16:

    m = deaton(quad='legendre')                   # the model object exactly as in Class 16
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import minimize_scalar, brentq


def quadrature_hermite(sigma, n):
    '''Gauss-Hermite nodes and weights for expectations of functions of a log-normal income'''
    x, w = np.polynomial.hermite.hermgauss(n)          # weight exp(-x^2) on the real line
    return np.exp(np.sqrt(2) * sigma * x), w / np.sqrt(np.pi)   # nodes y_k, weights sum to one

def quadrature_cdf(sigma, n):
    '''Gauss-Legendre on the CDF: nodes y_k = F^{-1}(q_k)'''
    x, w = np.polynomial.legendre.leggauss(n)          # Gauss-Legendre on [-1,1]
    return lognorm.ppf((x + 1) / 2, sigma), w / 2      # rescaled to [0,1], weights sum to one


class deaton():
    '''Stochastic consumption-savings model of Deaton (1991) with log-normal income'''

    def __init__(self, beta=0.9, R=1.05, sigma=0.5, Mbar=10, ngrid=100, nchgrid=250, nquad=10,
                 quad='hermite'):
        self.beta, self.R, self.sigma = beta, R, sigma   # discount factor, gross return, income std
        self.Mbar = Mbar                                  # upper bound of the state grid
        self.ngrid, self.nchgrid, self.nquad = ngrid, nchgrid, nquad
        self.eps = np.finfo(float).eps                    # smallest positive float
        self.grid = np.linspace(self.eps, Mbar, ngrid)    # state grid on (0, Mbar]
        self.share = np.linspace(self.eps, 1, nchgrid)    # choice grid as shares of wealth
        rules = {'hermite': quadrature_hermite,           # Gauss-Hermite, exact for normal log income
                 'legendre': quadrature_cdf}              # Gauss-Legendre on the CDF, as in Class 16
        if quad not in rules:
            raise ValueError(f"quad must be one of {list(rules)}, got {quad!r}")
        self.quad = quad
        self.quadp, self.quadw = rules[quad](sigma, nquad)   # income nodes y_k and weights
        assert np.isclose(self.quadw.sum(), 1)            # weights of an expectation sum to one

    def u(self, c):
        return np.log(c)                                  # utility

    def mu(self, c):
        return 1 / c                                      # marginal utility

    def imu(self, x):
        return 1 / x                                      # inverse marginal utility

    def next_wealth(self, M, c, y):
        return self.R * (M - c) + y                       # budget constraint

def interp(x, xp, fp):
    '''Linear interpolation, extrapolated linearly beyond the last grid point'''
    slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return np.where(x > xp[-1], fp[-1] + slope * (x - xp[-1]), np.interp(x, xp, fp))


def bellman_discretized(m, V0):
    '''Bellman operator with discretized choice, V0 is a vector of values on the state grid'''
    c = m.share[:, None, None] * m.grid[None, :, None]   # choices x states x 1, scaled to [0, M]
    M1 = m.next_wealth(m.grid[None, :, None], c, m.quadp[None, None, :])   # choices x states x nodes
    EV = interp(M1, m.grid, V0) @ m.quadw                 # expected value, choices x states
    X = m.u(c[:, :, 0]) + m.beta * EV                     # maximand of the Bellman equation
    i = np.argmax(X, axis=0)                              # optimal choice in every state
    j = np.arange(m.ngrid)
    return X[i, j], c[i, j, 0]

def solve_vfi(m, bellman, maxiter=500, tol=1e-4, callback=None):
    '''Value function iterations with the given Bellman operator, returns the policy'''
    V0 = m.u(m.grid)                                      # start from consuming everything
    for i in range(maxiter):
        V1, c1 = bellman(m, V0)
        err = np.max(np.abs(V1 - V0))
        if callback: callback(iter=i, value=V1, policy=c1, err=err)
        if err < tol: break
        V0 = V1
    else:
        raise RuntimeError('No convergence: maximum number of iterations achieved')
    return c1


def bellman_continuous(m, V0):
    '''Bellman operator with continuous choice, solved by Brent's method in every state'''
    def minus_maximand(c, M):
        EV = interp(m.next_wealth(M, c, m.quadp), m.grid, V0) @ m.quadw
        return -(m.u(c) + m.beta * EV)                    # minus: minimize_scalar minimizes
    V1, c1 = np.empty(m.ngrid), np.empty(m.ngrid)
    for j, M in enumerate(m.grid):
        if M <= m.grid[0]:
            c1[j] = M                                     # no room to optimize at the first point
        else:
            res = minimize_scalar(minus_maximand, args=(M,), method='bounded',
                                  bounds=(m.eps, M), options={'xatol': 1e-10})
            c1[j] = res.x
        V1[j] = -minus_maximand(c1[j], M)
    return V1, c1


def euler_residual(m, c, M, c0):
    '''LHS - RHS of the Euler equation at (M, c), with next period policy c0 on the grid'''
    M1 = m.next_wealth(np.atleast_1d(M)[:, None], np.atleast_1d(c)[:, None], m.quadp[None, :])
    c1 = np.maximum(interp(M1, m.grid, c0), m.eps)        # next period consumption, kept positive
    return m.mu(c) - m.beta * m.R * (m.mu(c1) @ m.quadw).squeeze()

def solve_timeiter(m, maxiter=500, tol=1e-4, callback=None):
    '''Time iterations: solve the Euler equation for c in every state, returns the policy'''
    c0 = m.grid.copy()                                    # start from consuming everything
    for i in range(maxiter):
        c1 = np.empty(m.ngrid)
        for j, M in enumerate(m.grid):
            if M <= m.grid[0] or euler_residual(m, M, M, c0) > 0:
                c1[j] = M                                 # constraint binds: consume everything
            else:
                c1[j] = brentq(lambda c: euler_residual(m, c, M, c0), m.eps, M, xtol=1e-12)
        err = np.max(np.abs(c1 - c0))
        if callback: callback(iter=i, policy=c1, err=err)
        if err < tol: break
        c0 = c1
    else:
        raise RuntimeError('No convergence: maximum number of iterations achieved')
    return c1


def accuracy(m, c, factor=10, rtol=1e-6):
    '''Mean absolute relative Euler residual of the policy c on a dense grid, interior points only'''
    M = np.linspace(m.grid[1], m.Mbar, m.ngrid * factor)  # dense grid, first point skipped
    cM = interp(M, m.grid, c)                             # interpolated policy
    er = euler_residual(m, cM, M, c) / m.mu(cM)           # relative Euler residual
    interior = cM < M * (1 - rtol)                        # drop the corner solutions c = M
    return np.mean(np.abs(er[interior]))


def simulate(m, c, M0, T=50, seed=None):
    '''Simulate T periods from initial wealth M0 (scalar or array) with the policy c on the grid'''
    rng = np.random.default_rng(seed)
    M0 = np.atleast_1d(M0).astype(float)
    M, C = np.empty((M0.size, T)), np.empty((M0.size, T))
    M[:, 0] = M0
    for t in range(T):
        C[:, t] = np.minimum(interp(M[:, t], m.grid, c), M[:, t])   # no borrowing
        if t < T - 1:
            y = lognorm.rvs(m.sigma, size=M0.size, random_state=rng)
            M[:, t + 1] = m.next_wealth(M[:, t], C[:, t], y)
    return M, C
