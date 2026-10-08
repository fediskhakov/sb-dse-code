"""Two-step CCP estimation of the Rust (1987) bus engine model.

The code of Class 13 (``session13-oct6/ccp_practice.ipynb``) collected into a module:
a first stage that estimates the conditional choice probabilities from the data, and
a second stage that is a static binary logit with an offset.

    import sys; sys.path.insert(0, '../session10-11')   # zurcher.py, nfxp.py, the data
    from nfxp import read_busdata, estim_zurcher
    from ccp import flexible_logit, ccp_second_stage
    est = estim_zurcher(read_busdata())                  # the model with the data attached
    theta, se, ll = ccp_second_stage(est, flexible_logit(est, degree=3))

    python ccp.py                                        # the same on Harold Zurcher's data

All functions take ``model``, an ``estim_zurcher`` object of ``nfxp.py`` (or any
``zurcher`` model with data attributes ``x``, ``d`` and ``N``), and work with the
replacement probability P(replace|x) on the whole grid.

First stage.  ``ccp_frequency`` is the bin estimator, undefined in the bins that are
never visited and zero in the bins without replacements, where the offset is not
defined.  ``flexible_logit`` is a logit in Chebyshev polynomials of mileage, positive
everywhere.

Second stage.  Under one-period finite dependence (replacement resets the state)

    v(x,keep) - v(x,replace) = RC - 0.001 c x + offset(x),
    offset(x) = -beta sum_x' log P(replace|x') [pi(x'|x,keep) - pi(x'|x,replace)],

so P(keep|x) is a static logit in (RC, c) with a known offset, estimated by
``logit_mle`` (Newton with step halving).
"""

import numpy as np
from numpy.polynomial import chebyshev

def logit_binary(z):
    '''P(y=1) = e^z / (1 + e^z) of the binary logit with utilities (0, z), de-maxed'''
    mx = np.maximum(0, z)                            # de-maxing constant: max of the two utilities
    return np.exp(z - mx) / (np.exp(-mx) + np.exp(z - mx))   # exponents of non-positive numbers

def logsum_binary(z):
    '''log(1 + e^z) = log(e^0 + e^z), the log-sum of utilities (0, z), de-maxed'''
    mx = np.maximum(0, z)
    return mx + np.log(np.exp(-mx) + np.exp(z - mx))

def fd_offset(model, P_rep):
    '''Future term of v(x,keep) - v(x,replace) under one-period finite dependence,
    computed from the CCPs P_rep = P(replace|x) on the grid:
    -beta * sum_x' log P(replace|x') [pi(x'|x,keep) - pi(x'|x,replace)]'''
    logP = np.log(P_rep)                             # -inf where P = 0: the offset is undefined there
    Pi = model.trpr                                  # keep: row x of the transition matrix
    return -model.beta * (Pi - Pi[0]) @ logP         # replace: row 0, a fresh engine, for every x

def ccp_frequency(model):
    '''Bin estimator of P(replace|x): replacements over visits, NaN where never visited.
    Returns the probabilities and the number of visits to each bin.'''
    visits = np.bincount(model.x, minlength=model.n)
    replacements = np.bincount(model.x, weights=model.d, minlength=model.n)
    with np.errstate(invalid='ignore'):              # 0/0 in the unvisited bins
        return replacements / visits, visits

def logit_mle(X, y, offset=0.0, tol=1e-10, maxiter=100, callback=None):
    '''Binary logit P(y=1|X) = Lambda(X b + offset) by Newton's method with step halving,
    with given tolerance and number of iterations.
    Callback function is invoked at each iteration if given.
    Returns b, standard errors from the inverse Hessian, and the log-likelihood.
    '''
    def loglik(b):
        z = X @ b + offset
        return np.sum(y * z - logsum_binary(z))    # log Lambda(z) = z - log(1 + e^z)
    b0 = np.zeros(X.shape[1])
    for i in range(maxiter):
        p = logit_binary(X @ b0 + offset)              # Lambda(z)
        g = X.T @ (y - p)                              # score
        H = (X * (p * (1 - p))[:, None]).T @ X         # minus the Hessian, positive definite
        step = np.linalg.solve(H, g)                   # Newton step
        while loglik(b0 + step) < loglik(b0) and np.abs(step).max() > tol:
            step /= 2                                  # plain Newton overshoots from b = 0
        b1 = b0 + step
        err = np.abs(b1 - b0).max()
        if callback is not None: callback(err=err, b0=b0, b1=b1, iter=i)
        if err < tol: break
        b0 = b1
    else:
        raise RuntimeError('Failed to converge in %d iterations' % maxiter)
    return b1, np.sqrt(np.diag(np.linalg.inv(H))), loglik(b1)

def flexible_logit(model, degree=3, full=False, keep=False):
    '''Flexible logit first stage: P(replace|x) on the whole grid from a logit in
    Chebyshev polynomials of mileage, which extrapolates to the bins without replacements.
    With keep=True returns P(keep|x) instead, computed from the logit directly, not as 1 - P.
    With full=True also returns the coefficients, their standard errors and the log-likelihood.'''
    basis = lambda x: chebyshev.chebvander(2 * x / model.n - 1, degree)   # grid mapped to [-1, 1]
    b, se, ll = logit_mle(basis(model.x), model.d)
    z = basis(model.grid) @ b                             # log odds of replacement
    P = logit_binary(-z if keep else z)                   # Lambda(-z) = 1 - Lambda(z), without the cancellation
    return (P, b, se, ll) if full else P

def ccp_second_stage(model, P_rep):
    '''Second stage: P(keep|x) = Lambda(RC - 0.001 c x + offset(x)), a static logit in
    (RC, c) with the future term computed from the first-stage CCPs P_rep = P(replace|x).
    Returns (RC, c), standard errors, log-likelihood.'''
    offset = fd_offset(model, P_rep)
    X = np.column_stack([np.ones(model.N), -0.001 * model.x])   # coefficients: RC and c
    return logit_mle(X, 1 - model.d, offset[model.x])           # y = 1 for keep

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'session10-11'))
    from nfxp import read_busdata, estim_zurcher

    data = read_busdata()                        # bus groups 1-4, 175 bins, 450 000 miles
    est = estim_zurcher(data)                    # the model with the data attached

    P_hat = flexible_logit(est, degree=3)        # first stage: Chebyshev logit of degree 3
    theta, se, ll = ccp_second_stage(est, P_hat) # second stage: static logit with the offset

    res = estim_zurcher(data).estimate(full=False)   # NFXP partial likelihood, same p, fresh model

    print(f'{"estimator":<20}{"RC":>8}{"s.e.":>8}{"c":>8}{"s.e.":>8}{"loglik":>10}')
    print(f'{"two-step CCP":<20}{theta[0]:8.3f}{se[0]:8.3f}{theta[1]:8.3f}{se[1]:8.3f}{ll:10.2f}')
    print(f'{"NFXP":<20}{res.theta[0]:8.3f}{res.se[0]:8.3f}{res.theta[1]:8.3f}{res.se[1]:8.3f}'
          f'{res.stages[0]["loglik_choice"]:10.2f}')
