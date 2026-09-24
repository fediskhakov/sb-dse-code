"""Nested fixed point (NFXP) estimation of the Rust (1987) bus engine model.

The three things an estimator needs, on top of the model in ``zurcher.py``:
read the data, compute the likelihood, maximize it.

    from nfxp import read_busdata, estim_zurcher
    est = estim_zurcher(read_busdata())   # the model of zurcher.py with the data attached
    result = est.estimate()               # est.RC, est.c, est.p now hold the estimates
    print(result)

    python nfxp.py                        # the same on Harold Zurcher's data

The estimator is a subclass of the model: it inherits the grid, the transition
matrix, the Bellman operator and the solvers, and adds the data, the likelihood,
its score and the outer loop.  The parameters of the object are the current point
of the outer loop, so after ``estimate()`` the same object is ready for the
counterfactuals (see ``replicate_rust1987.ipynb``).

Parameter vector.  theta = (RC, c) for the partial likelihood and
theta = (RC, c, p_0, ..., p_{J-1}) for the full likelihood, where J is the
largest mileage increment observed in the data and p_J = 1 - sum(p) is the
residual probability.  The maintenance cost is 0.001 * c * x for mileage bin x.

Likelihood.  Every bus-month contributes the log choice probability and the
log probability of the mileage increment that led to the current state::

    l_i = log P(d_i | x_i; theta) + log p_{dx_i}

Analytical score.  With v0 = -cost + beta*EV and v1 = -RC - cost[0] + beta*EV[0]
the choice part of the score is (1 - d - P(keep|x)) * d(v0(x) - v1)/dtheta,
and the derivative of the fixed point comes from the implicit function
theorem, dEV/dtheta = (I - Gamma')^{-1} dGamma/dtheta, using the Fréchet
derivative Gamma' that the Newton-Kantorovich step already computes.

Sources: Rust (1987), the NFXP manual (Rust 2000), and the DSE/UiO lab code by
Bertel Schjerning and Fedor Iskhakov, rewritten around the model class of
the course.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from zurcher import zurcher

DATAFILE = Path(__file__).parent / 'busdata1234.csv'

# Inner loop settings.  The fixed point is always finished with Newton-Kantorovich
# steps, whose stopping rule is trustworthy: NK converges quadratically, so a step
# smaller than tol means the true error is far smaller still.  A contraction step
# of size tol, in contrast, can still be tol/(1-beta) away from the fixed point.
# tol is relative to the scale of EV, because the NK linear system has condition
# number of order 1/(1-beta) and its precision floor scales with |EV|.
SOLVER = dict(
    tol=1e-10,          # relative tolerance of the final NK steps
    nk_maxiter=50,      # NK steps before declaring a divergence
    cold=dict(tol=1e-6, maxiter=2000, sa_min=5, sa_max=50, switch_tol=1e-3),  # poly-algorithm from zero
)


# ============================================================================ data

def read_busdata(path=DATAFILE, groups=(1, 2, 3, 4), n=175, max_miles=450_000):
    '''Read Zurcher's maintenance records and discretize mileage on an n-point grid.

    busdata1234.csv has one row per bus-month and no header.  Columns:
      0 bus id           
      1 bus group (1-4)      
      2 year (75-85)     
      3 month (1-12)
      4 replacement flag: 1 in the month the engine was replaced
      5 mileage since the last replacement, previous month
      6 mileage since the last replacement, this month (reset in a replacement month)
      7 odometer reading (cumulative)         
      8 change in column 6

    The decision is recorded one row later than the state it was taken in: the
    flag in row t+1 says that the bus that arrived with mileage m_t in row t was
    replaced, and column 6 in row t+1 already holds the mileage on the new engine.
    So d_t is the flag of the next row, and each observation is a state, the
    decision taken there, and the transition that led into that state; the first
    month of every bus is dropped because it has no transition, and the decision
    in a bus's last month, which is not observed, is recorded as keep.

    The discretization follows Rust's original code: the cell index
    k = ceil(m * n / max_miles) runs from 1, and is used as a 1-based index into
    the grid 0, ..., n-1.  Hence the decision is evaluated at grid value x = k - 1,
    and the mileage change is the change in k: for a kept engine the transition
    runs from x_from = x_{t-1} to x_to = x_t; after a replacement it is recorded
    as the increment k_t from grid value 0, that is x_to = x_t + 1, one cell more
    than the model's own reset would produce.  This is the known off-by-one of
    Rust's data, kept because it is what the paper computes: with it, the
    estimator reproduces Rust (1987) Tables IX and X.

    Returns a DataFrame with columns bus, group, year, month, miles, x, x_from,
    x_to, d, dx = x_to - x_from, one row per bus-month, and n, max_miles and
    groups in its attrs.
    '''
    raw = np.loadtxt(path, delimiter=',')
    df = pd.DataFrame({
        'bus': raw[:, 0].astype(int),
        'group': raw[:, 1].astype(int),
        'year': raw[:, 2].astype(int),
        'month': raw[:, 3].astype(int),
        'replaced': raw[:, 4].astype(int),
        'miles': raw[:, 6],
    })
    df = df.sort_values(['bus', 'year', 'month'], kind='stable').reset_index(drop=True)
    g = df.groupby('bus', sort=False)
    df['d'] = g['replaced'].shift(-1).fillna(0).astype(int)            # decision at x_t
    df['k'] = np.ceil(df['miles'] * n / max_miles).astype(int)          # Rust's cell index, from 1
    k, k_prev = df['k'], g['k'].shift(1)                                # previous month of the same bus
    df['x'] = np.minimum(k - 1, n - 1)                                  # grid value the decision is evaluated at
    df['x_from'] = np.where(df['replaced'] == 1, 0, np.minimum(k_prev - 1, n - 1))   # where the transition started
    df['x_to'] = np.minimum(np.where(df['replaced'] == 1, k, k - 1), n - 1)   # where it landed, as recorded
    df = df[k_prev.notna()]                                             # drop first month
    df = df[df['group'].isin(groups)]
    df = df.astype({'x_from': int, 'x_to': int})
    df['dx'] = df['x_to'] - df['x_from']
    assert (df['dx'] >= 0).all(), 'negative mileage change: rows out of order or a replacement not flagged'
    out = df[['bus', 'group', 'year', 'month', 'miles', 'x', 'x_from', 'x_to', 'd', 'dx']].reset_index(drop=True)
    out.attrs.update(n=n, max_miles=max_miles, groups=tuple(groups))
    return out


def describe_data(data):
    '''Print what the estimator will see'''
    n, max_miles = data.attrs.get('n', '?'), data.attrs.get('max_miles', '?')
    print(f'Bus groups {data.attrs.get("groups", "?")}: {data.bus.nunique()} buses, '
          f'{len(data)} bus-months, {int(data.d.sum())} engine replacements')
    print(f'Mileage grid: n = {n} bins of {max_miles / n / 1000:.2f} thousand miles; '
          f'highest bin observed {data.x.max()}')
    freq = data.dx.value_counts().sort_index()
    print('Monthly mileage increment in bins (dx), frequencies:')
    print('  ' + '  '.join(f'{j}: {c} ({c / len(data):.4f})' for j, c in freq.items()))
    miles_at_replacement = data.loc[data.d == 1, 'miles']
    print(f'Mileage at replacement: mean {miles_at_replacement.mean() / 1000:.0f}k, '
          f'min {miles_at_replacement.min() / 1000:.0f}k, max {miles_at_replacement.max() / 1000:.0f}k')


# ======================================================================= estimator

@dataclass
class Result:
    '''Output of estim_zurcher.estimate()'''
    names: list
    theta: np.ndarray
    se: np.ndarray
    vcov: np.ndarray
    loglik: float
    N: int
    method: str
    converged: bool
    time: float
    n_solver_calls: int
    n_inner_iter: int
    beta: float
    n: int
    stages: list = field(default_factory=list)   # scipy/BHHH results of each stage

    def table(self):
        '''Estimates and standard errors, one line per parameter'''
        lines = [f'{"parameter":<10}{"estimate":>12}{"s.e.":>12}']
        for name, th, se in zip(self.names, self.theta, self.se):
            lines.append(f'{name:<10}{th:>12.5f}{se:>12.5f}')
        return '\n'.join(lines)

    def __str__(self):
        head = (f'NFXP estimates, beta = {self.beta}, n = {self.n}, N = {self.N} bus-months\n'
                f'method {self.method}, converged {self.converged}, {self.time:.3f} s, '
                f'{self.n_solver_calls} inner solves, {self.n_inner_iter} inner iterations\n')
        return head + self.table() + f'\nlog-likelihood {self.loglik:.4f}'


class estim_zurcher(zurcher):
    '''Nested fixed point maximum likelihood estimator for the Zurcher model.

    The estimator is the model with data attached: it inherits the grid, the
    transition matrix, the Bellman operator and the solvers from zurcher, and adds
    the likelihood, its score and the outer loop.  RC, c and p are the current point
    of the outer loop; after estimate() they hold the estimates and the object is
    ready for the counterfactuals.
    '''

    def __init__(self, data,
                 n=None,             # grid points; default: the grid the data were discretized on
                 beta=0.9999,        # discount factor, fixed in estimation
                 solver=None,        # overrides of the inner loop settings in SOLVER
                 warm_start=True,    # start the inner loop from the previous solution
                 J=None,             # number of free transition probabilities
                 **params):          # RC, c, p as in zurcher; p defaults to the increment frequencies
        '''Attach the data, then build the model.

        data is a DataFrame from read_busdata() with columns x, d, x_from, x_to, dx.
        J counts the free transition probabilities (increments 0..J, the last one
        residual); by default the largest increment seen in the data, which is right
        for Rust's data.  Pass it explicitly for a sample in which the largest
        category does not occur.  p defaults to the frequencies of the increments,
        the closed form estimate of stage 1, so that len(p) == J from the start and
        the object is consistent before estimate() is ever called.'''
        self.data = data
        self.x = data['x'].to_numpy()               # state the decision is taken at
        self.d = data['d'].to_numpy()               # 1 = replace
        self.x_from = data['x_from'].to_numpy()     # the transition that led into x ...
        self.x_to = data['x_to'].to_numpy()         # ... as recorded, see read_busdata()
        self.dx = data['dx'].to_numpy()             # = x_to - x_from, the mileage increment in bins
        self.N = self.x.size
        self.J = int(self.dx.max()) if J is None else int(J)
        self.names = ['RC', 'c'] + [f'p{j}' for j in range(self.J)]
        self.max_miles = data.attrs.get('max_miles', 450_000)   # only to label the grid in miles
        assert 0 <= self.dx.min() and self.dx.max() <= self.J, 'mileage change outside the transition categories 0..J'
        assert n is not None or 'n' in data.attrs, 'pass n: the data do not say which grid they were discretized on'
        params.setdefault('p', self.transition_freq())
        super().__init__(n=data.attrs['n'] if n is None else n, beta=beta, **params)   # builds grid and trpr
        assert len(self.p) == self.J, 'one transition probability per free increment category'
        assert 0 <= self.x.min() and self.x.max() < self.n, 'data contain mileage bins outside the grid of the model'
        assert 0 <= self.x_from.min() and self.x_to.max() < self.n, 'transitions outside the grid of the model'
        self.solver = dict(SOLVER, **(solver or {}))
        self.warm_start = warm_start
        self.reset()

    def __repr__(self):
        return f'NFXP estimator of the Rust model on {self.N} bus-months (id={id(self)})'

    def reset(self):
        '''Forget the warm start and zero the inner loop counters'''
        self.ev = None                          # last inner solution, used as warm start
        self.n_solver_calls = 0
        self.n_inner_iter = 0

    # ---------------------------------------------------------------- parameters
    @property
    def theta(self):
        '''Parameters as the vector (RC, c, p_0, ..., p_{J-1}) the likelihood is written in'''
        return np.array([self.RC, self.c] + self.p)

    @theta.setter
    def theta(self, value):
        '''Assigning (RC, c) moves the cost parameters, (RC, c, p) all of them'''
        value = np.asarray(value, dtype=float)
        assert value.size in (2, 2 + self.J), f'theta must have 2 or {2 + self.J} elements'
        self.RC, self.c = float(value[0]), float(value[1])
        if value.size > 2:
            self.p = value[2:]

    def transition_freq(self):
        '''Transition probabilities from the increment frequencies: the closed form MLE
        away from the top of the grid, and the starting point of the full likelihood.
        An unobserved category gets half an observation so the start is inside the simplex.'''
        counts = np.bincount(self.dx, minlength=self.J + 1).astype(float)
        counts = np.maximum(counts, 0.5)
        return counts[:self.J] / counts.sum()

    def feasible(self, theta):
        '''The transition probabilities in theta must form a distribution: the likelihood
        is -inf outside, and the outer loop must not step there'''
        p = theta[2:]
        return bool(np.all(p > 0) and p.sum() < 1)

    def snapshot(self):
        '''Parameters and warm start, so a diagnostic can leave the estimator as it found it'''
        return self.theta, None if self.ev is None else self.ev.copy()

    def restore(self, snap):
        '''Undo what happened since snapshot()'''
        self.theta, self.ev = snap

    def thousand_miles(self, x):
        '''Grid value x in thousands of miles, for tables and plots'''
        return x * self.max_miles / self.n / 1000

    # ---------------------------------------------------------------- inner loop
    def solve_fixed_point(self, ev0=None, callback=None):
        '''Solve the model at its current parameters with the settings in self.solver:
        NK steps from ev0 if given, otherwise the poly-algorithm from zero followed by
        NK steps; if NK diverges from a poor warm start, start cold.  The NK tolerance
        is relative to the scale of EV: set from the starting point and, if the solution
        lives on a smaller scale, the last steps are repeated at the tolerance that
        scale demands.  Returns EV and P(keep|x).'''
        tol, nk_maxiter, cold = self.solver['tol'], self.solver['nk_maxiter'], self.solver['cold']
        if ev0 is None:
            ev0, _ = self.solve_poly(callback=callback, **cold)
        for attempt in range(2):
            try:
                with np.errstate(all='ignore'):       # a diverging NK step overflows before it raises
                    tol0 = tol * max(1.0, np.abs(ev0).max())
                    ev, pk = self.solve_nk(ev0=ev0, tol=tol0, maxiter=nk_maxiter, callback=callback)
                    tol1 = tol * max(1.0, np.abs(ev).max())
                    if tol1 < tol0:
                        ev, pk = self.solve_nk(ev0=ev, tol=tol1, maxiter=nk_maxiter, callback=callback)
                return ev, pk
            except RuntimeError:
                if attempt == 1:
                    raise
                ev0, _ = self.solve_poly(callback=callback, **cold)   # cold restart

    def solve(self, theta=None):
        '''Move to theta if given and solve the model, warm started from the previous
        solution when allowed.  Returns EV, P(keep|x) and the Fréchet derivative.'''
        if theta is not None:
            self.theta = theta
        ev0 = self.ev if (self.warm_start and self.ev is not None and self.ev.size == self.n) else None

        def count(**kw):
            self.n_inner_iter += 1

        ev, pk = self.solve_fixed_point(ev0=ev0, callback=count)
        self.n_solver_calls += 1
        self.ev = ev
        return self.bellman(ev, deriv=True)   # ev, pk, dev at the fixed point

    def choice_values(self, ev):
        '''Choice specific values v(x, keep), v(x, replace) and their logsum at a solution.
        The Bellman operator of the parent computes them but does not return them.'''
        cost = 0.001 * self.c * self.grid
        v0 = -cost + self.beta * ev
        v1 = -cost[0] - self.RC + self.beta * ev[0]
        return v0, v1, np.logaddexp(v0, v1)

    # ---------------------------------------------------------------- likelihood
    def loglik_choice_obs(self, theta):
        '''Log choice probability log P(d | x) of every bus-month at theta'''
        theta = np.asarray(theta, dtype=float)
        if not self.feasible(theta):
            return np.full(self.N, -np.inf)
        ev, _, _ = self.solve(theta)
        v0, v1, L = self.choice_values(ev)
        logP0, logP1 = v0 - L, v1 - L           # log probabilities directly: never log of a probability
        return np.where(self.d == 0, logP0[self.x], logP1[self.x])

    def loglik_transition_obs(self):
        '''Log probability of the observed mileage increment of every bus-month, read off
        the model's own transition matrix: p_dx away from the top of the grid, and the
        cumulative probability where the top bin absorbs'''
        with np.errstate(divide='ignore'):     # a zero probability is a legitimate -inf
            return np.log(self.trpr[self.x_from, self.x_to])

    def loglik_obs(self, theta):
        '''Log-likelihood contribution of every bus-month, choice part plus transition part'''
        choice = self.loglik_choice_obs(theta)     # moves to theta first, so trpr is at theta below
        return choice + self.loglik_transition_obs()

    def loglik(self, theta):
        '''Log-likelihood, summed over bus-months'''
        return float(np.sum(self.loglik_obs(theta)))

    def loglik_choice(self, theta):
        '''The choice part of the log-likelihood alone.  With p held fixed this is the
        partial likelihood of stage 2 up to a constant, and it is the number the
        reference implementations report for that stage.'''
        return float(np.sum(self.loglik_choice_obs(theta)))

    # ---------------------------------------------------------------- score
    def score(self, theta):
        '''N x K matrix of scores, K = theta.size, with the derivative of the fixed
        point from the implicit function theorem'''
        theta = np.asarray(theta, dtype=float)
        K = theta.size
        if not self.feasible(theta):             # the likelihood is -inf there: no direction to report
            return np.zeros((self.N, K))
        ev, pk, dev = self.solve(theta)
        _, _, L = self.choice_values(ev)
        # implicit function theorem on the fixed point EV = Gamma(EV)
        dEV = np.linalg.solve(np.eye(self.n) - dev, self.dbellman_dtheta(pk, L, K))
        # choice part: (1 - d - P(keep|x)) times the derivative of v(x, keep) - v(x, replace)
        dv = self.beta * (dEV[self.x, :] - dEV[0, :])
        dv[:, 0] += 1.0
        dv[:, 1] += -0.001 * self.x
        S = (1 - self.d - pk[self.x])[:, None] * dv
        if K > 2:
            S[:, 2:] += self.dlog_transition()
        return S

    def dbellman_dtheta(self, pk, L, K):
        '''Derivative of the Bellman operator w.r.t. the first K parameters, EV held fixed, n x K'''
        n, grid, J = self.n, self.grid, self.J
        dG = np.zeros((n, K))
        dG[:, 0] = -(self.trpr @ (1 - pk))                     # RC: replace value moves by -1
        dG[:, 1] = -0.001 * (self.trpr @ (pk * grid))          # c: keep value moves by -0.001 x
        for j in range(K - 2):
            # dPi/dp_j: +1 in column min(i+j, n-1), -1 in column min(i+J, n-1) of row i
            dG[:, 2 + j] = L[np.minimum(grid + j, n - 1)] - L[np.minimum(grid + J, n - 1)]
        return dG

    def dlog_transition(self):
        '''Derivative of log Pi[x_from, x_to] w.r.t. p_0, ..., p_{J-1}, N x J: the derivative
        D_j of the matrix over its entry.  Away from the top bin this is +1/p_dx when
        dx == j and -1/p_J when dx == J.'''
        n, J = self.n, self.J
        pr = self.trpr[self.x_from, self.x_to]
        col_J = np.minimum(self.x_from + J, n - 1)
        D = np.empty((self.N, J))
        for j in range(J):
            col_j = np.minimum(self.x_from + j, n - 1)
            D[:, j] = ((self.x_to == col_j).astype(float) - (self.x_to == col_J)) / pr
        return D

    # ---------------------------------------------------------------- outer loop
    def bhhh(self, theta0, tol=1e-10, maxiter=200, verbose=False):
        '''Berndt-Hall-Hall-Hausman: Newton steps with the outer product of the scores
        as the Hessian and a backtracking line search.  Stops when the predicted gain
        of the next step, g'(S'S)^{-1}g, drops below tol.  The likelihood is very flat
        in RC (a step of 0.0002 costs 5e-8 in log-likelihood), so tol has to be this
        small for the fourth decimal of the estimates to be settled.

        The outer product of the scores equals the negative Hessian only in expectation
        and only if the model is right, so near the optimum the BHHH direction can be
        poor and the iterations crawl (the myopic model beta = 0 is a case).  When the
        criterion stops falling by at least half per iteration the step is taken along
        the Newton direction with the numerical Hessian of the log-likelihood instead.'''
        theta = np.asarray(theta0, dtype=float).copy()
        ll = self.loglik(theta)
        converged = False
        crit_prev = np.inf
        for it in range(1, maxiter + 1):
            S = self.score(theta)
            g = S.sum(axis=0)
            step = np.linalg.solve(S.T @ S, g)
            crit = g @ step                                     # predicted gain of the step
            if crit > 0.5 * crit_prev and crit < 1e-2:          # crawling near the optimum
                step_newton = np.linalg.solve(-self.numerical_hessian(theta), g)
                if g @ step_newton > 0:                         # only if it is an ascent direction
                    step, crit = step_newton, g @ step_newton
            crit_prev = crit
            theta1, ll1, lam = self.line_search(theta, step, ll)
            if lam == 0:
                # no step improves the likelihood: its evaluation is at the precision floor
                # of the inner loop; converged if the predicted gain crit is negligible too
                converged = bool(crit < 1e4 * tol)
                break
            theta, ll = theta1, ll1
            if verbose:
                print(f'  BHHH {it:3d}  loglik {ll:14.6f}  crit {crit:10.3e}  step {lam:g}')
            if crit < tol:
                converged = True
                break
        return dict(x=theta, fun=-ll / self.N, nit=it, success=converged, message='BHHH')

    def line_search(self, theta, step, ll):
        '''Backtracking: halve the step until the log-likelihood does not fall.
        Returns the new point, its log-likelihood and the step length, 0 if none worked.'''
        lam = 1.0
        while lam > 1e-10:
            theta1 = theta + lam * step
            ll1 = self.loglik(theta1)
            if np.isfinite(ll1) and ll1 >= ll:
                return theta1, ll1, lam
            lam /= 2
        return theta, ll, 0.0

    # scipy minimizes: the negative mean log-likelihood, its gradient and the BHHH Hessian
    def objective(self, theta):
        ll = self.loglik(theta)
        return -ll / self.N if np.isfinite(ll) else 1e10    # a finite penalty outside the simplex

    def objective_grad(self, theta):
        return -self.score(theta).mean(axis=0)

    def objective_hess(self, theta):
        S = self.score(theta)
        return S.T @ S / self.N

    def maximize(self, theta0, method='bhhh', verbose=False):
        '''Dispatch to the chosen optimizer; returns (theta, converged, raw result)'''
        theta0 = np.asarray(theta0, dtype=float)
        if method == 'bhhh':
            res = self.bhhh(theta0, verbose=verbose)
            return res['x'], res['success'], res
        # gtol applies to the gradient of the mean log-likelihood, 1/N of the score sum
        if method == 'trust-ncg':                      # analytical gradient, BHHH Hessian
            res = optimize.minimize(self.objective, theta0, jac=self.objective_grad, hess=self.objective_hess,
                                    method='trust-ncg', options=dict(gtol=1e-6))
        elif method == 'bfgs':                         # analytical gradient, no Hessian
            res = optimize.minimize(self.objective, theta0, jac=self.objective_grad, method='BFGS',
                                    options=dict(gtol=1e-6))
        elif method == 'bfgs-numerical':               # finite difference gradient
            res = optimize.minimize(self.objective, theta0, method='BFGS', options=dict(gtol=1e-6))
        elif method == 'nelder-mead':                  # derivative free
            res = optimize.minimize(self.objective, theta0, method='Nelder-Mead',
                                    options=dict(xatol=1e-8, fatol=1e-12, maxfev=20000))
        else:
            raise ValueError(f'unknown method {method}')
        return res.x, bool(res.success), res

    def estimate(self, theta0=(0.0, 0.0), method='bhhh', full=True, verbose=False):
        '''Rust's three stages: transition probabilities from frequencies, then (RC, c)
        by partial likelihood with p fixed, then all parameters jointly.  The object
        ends up at the estimates; theta0 is the starting point for (RC, c).'''
        t0 = time.perf_counter()
        self.reset()
        stages = []
        # stage 1: closed form
        self.p = self.transition_freq()
        # stage 2: partial likelihood
        theta, ok, res = self.maximize(np.asarray(theta0, dtype=float)[:2], method, verbose)
        res['loglik_choice'] = self.loglik_choice(theta)   # the value of the partial likelihood
        stages.append(res)
        # stage 3: full likelihood, from the stage 1 and 2 estimates
        if full:
            theta, ok, res = self.maximize(np.append(theta, self.p), method, verbose)
            stages.append(res)
        self.theta = theta
        S = self.score(theta)
        vcov = np.linalg.inv(S.T @ S)                      # the inverse is wanted for itself here
        return Result(names=self.names[:theta.size], theta=theta, se=np.sqrt(np.diag(vcov)),
                      vcov=vcov, loglik=self.loglik(theta), N=self.N, method=method,
                      converged=ok, time=time.perf_counter() - t0,
                      n_solver_calls=self.n_solver_calls, n_inner_iter=self.n_inner_iter,
                      beta=self.beta, n=self.n, stages=stages)

    # ---------------------------------------------------------------- inference
    def numerical_hessian(self, theta, h=1e-5):
        '''Hessian of the log-likelihood by central differences of the analytical gradient.
        The step in the probability directions is capped at a quarter of the distance to
        the nearest edge of the simplex, the residual probability included, so no
        difference straddles the boundary where the score is switched off.'''
        theta = np.asarray(theta, dtype=float)
        K = theta.size
        steps = np.full(K, h)
        if K > 2:
            room = min(theta[2:].min(), 1 - theta[2:].sum())
            steps[2:] = min(h, 0.25 * room)
        snap = self.snapshot()
        H = np.zeros((K, K))
        for k in range(K):
            e = np.zeros(K)
            e[k] = steps[k]
            H[:, k] = (self.score(theta + e).sum(0) - self.score(theta - e).sum(0)) / (2 * steps[k])
        self.restore(snap)
        return (H + H.T) / 2

    def standard_errors(self, theta):
        '''Three estimates of the covariance matrix: outer product of the scores (BHHH),
        inverse negative Hessian, and the sandwich that combines them.'''
        theta = np.asarray(theta, dtype=float)
        S = self.score(theta)
        opg = S.T @ S
        H = self.numerical_hessian(theta)
        v_opg = np.linalg.inv(opg)
        v_hess = np.linalg.inv(-H)
        v_sand = v_hess @ opg @ v_hess
        return dict(bhhh=v_opg, hessian=v_hess, sandwich=v_sand)


if __name__ == '__main__':
    data = read_busdata()
    describe_data(data)
    print()
    est = estim_zurcher(data)
    result = est.estimate(verbose=True)
    print(result)
    print('Partial likelihood stage (p fixed at frequencies):',
          np.round(result.stages[0]['x'], 4), 'in', result.stages[0]['nit'], 'BHHH iterations,',
          f'choice log-likelihood {result.stages[0]["loglik_choice"]:.5f}')
