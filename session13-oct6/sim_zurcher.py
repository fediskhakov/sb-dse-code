"""Simulation and counterfactual demand for the Rust (1987) bus engine model.

Functions of any solved ``zurcher`` model, independent of the estimator: the
stationary distribution of mileage under a policy, a simulated panel of buses in the
format of the data, and the demand curve for engine replacement as a function of
the replacement cost.  Run the file for a demonstration at Rust's parameters::

    python sim_zurcher.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Absolute tolerance for the model's poly-algorithm.  EV is of the order of 1e3, and the
# solver's default 1e-10 is below what floating point can resolve at that scale for
# large RC; 1e-8 converges everywhere and changes the results below by 1e-12.
TOL = 1e-8


def ergodic_distribution(model, pk):
    '''Stationary distribution of mileage under the policy P(keep|x) = pk.

    The chain moves by the keep transition with probability pk and by the
    replace transition (the first row of trpr, a fresh engine) otherwise.
    Returns q(x), q(x, keep) and q(x, replace).
    '''
    n = model.n
    P = pk[:, None] * model.trpr + (1 - pk)[:, None] * model.trpr[0][None, :]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0                     # replace one redundant equation by the adding-up constraint
    b = np.zeros(n)
    b[-1] = 1.0
    q = np.linalg.solve(A, b)
    q = np.maximum(q, 0)
    q /= q.sum()
    return q, q * pk, q * (1 - pk)


def simulate(model, N=100, T=120, seed=None, pk=None, x0=None):
    '''Simulate N buses for T months from any zurcher model, solved with the model's own
    poly-algorithm unless the policy pk is given.

    The initial mileage is drawn from the ergodic distribution unless x0 gives a
    bin to start every bus from (x0 = 0: new engines, like the buses in the data).
    Returns a DataFrame with the columns the estimator reads, with exactly the
    meaning they have in read_busdata(): bus, t, x (state), x_from (where the
    transition into x started: the previous state, or 0 after a replacement),
    d (1 = replace), dx = x - x_from.  Like the data, dx is the observed change of
    the discretized state, so the top bin censors it.  The first month has no
    transition and is dropped, as in the data.
    '''
    rng = np.random.default_rng(seed)
    if pk is None:
        _, pk = model.solve_poly(tol=TOL)
    pfull = np.append(model.p, 1 - sum(model.p))
    if x0 is None:
        q, _, _ = ergodic_distribution(model, pk)
        x = rng.choice(model.n, size=N, p=q)
    else:
        x = np.full(N, int(x0))
    X, XF, D = np.empty((T, N), int), np.empty((T, N), int), np.empty((T, N), int)
    x_from = np.full(N, -1)                                 # undefined in the first month
    for t in range(T):
        d = rng.random(N) < 1 - pk[x]                       # replace this month
        j = rng.choice(pfull.size, size=N, p=pfull)         # mileage increment, in bins
        X[t], XF[t], D[t] = x, x_from, d
        x_from = np.where(d, 0, x)                          # mileage restarts after a replacement
        x = np.minimum(x_from + j, model.n - 1)             # the top bin absorbs
    bus = np.tile(np.arange(N), (T, 1))
    tt = np.tile(np.arange(T), (N, 1)).T
    df = pd.DataFrame({'bus': bus.ravel(order='F'), 't': tt.ravel(order='F'),
                       'x': X.ravel(order='F'), 'x_from': XF.ravel(order='F'), 'd': D.ravel(order='F')})
    df = df[df.t > 0].reset_index(drop=True)
    df['x_to'] = df['x']                                    # the model's own transitions land on the state
    df['dx'] = df['x_to'] - df['x_from']
    df.attrs.update(n=model.n, max_miles=450_000 * model.n / 175, groups='simulated')   # bins of Rust's size
    return df


def demand_curve(model, RC_grid):
    '''Expected annual engine replacements per bus as a function of the replacement cost,
    for any zurcher model; its other parameters stay where they are'''
    saved_RC = model.RC
    demand = np.empty(RC_grid.size)
    ev = None
    for i, rc in enumerate(RC_grid):
        model.RC = rc
        ev, pk = model.solve_poly(ev0=ev, tol=TOL)          # warm start along the grid
        _, _, q_r = ergodic_distribution(model, pk)
        demand[i] = 12 * q_r.sum()
    model.RC = saved_RC
    return demand


def demand_elasticity(model, RC, h=0.01):
    '''Elasticity of annual replacement demand with respect to RC at a point'''
    d = demand_curve(model, np.array([RC * (1 - h), RC * (1 + h)]))
    return (d[1] - d[0]) / (2 * h) / d.mean()


def demand_axes(title):
    '''Empty figure for demand curves in dollars'''
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel('replacement cost, dollars')
    ax.set_ylabel('expected engine replacements per bus per year')
    ax.set_title(title)
    ax.set_xlim(0, 12000)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    return fig, ax


if __name__ == '__main__':
    from zurcher import zurcher
    model = zurcher()                                       # Rust's Table X parameters
    panel = simulate(model, N=100, T=120, seed=1)
    print(f'Simulated {panel.bus.nunique()} buses x {panel.t.max()} months: '
          f'{len(panel)} bus-months, {int(panel.d.sum())} replacements, '
          f'monthly replacement rate {panel.d.mean():.4f}')
    RC_grid = np.linspace(1, 20, 39)
    fig, ax = demand_axes('Demand for engine replacement, one bus')
    ax.set_xlabel('replacement cost, utility units')
    ax.set_xlim(RC_grid[0], RC_grid[-1])
    ax.plot(RC_grid, demand_curve(model, RC_grid), lw=2)
    ax.axvline(model.RC, ls='--', color='gray')
    print(f'Demand elasticity with respect to RC at RC = {model.RC}: {demand_elasticity(model, model.RC):.3f}')
    plt.show()
