"""Rust (1987) bus engine replacement model: model class and three solvers.

The model of Class 9 in one importable file.  It is the class shown in the
lecture notes (https://dse.iskh.me/zurcher) with two additions that the NFXP
estimator in ``nfxp.py`` needs:

* ``p`` is a property, so assigning new transition probabilities rebuilds the
  transition matrix, exactly like assigning ``n`` rebuilds the grid;
* every solver takes an optional starting point ``ev0`` (default: zeros), so
  the estimator can warm-start the inner loop from the previous solution.

Notation follows the notes.  The state is the mileage bin index ``x`` on the
grid ``0, ..., n-1``, the maintenance cost is linear in it, ``0.001 * c * x``,
and the fixed point is computed in expected value function space::

    EV = trpr @ logsum(-cost + beta*EV, -RC - cost[0] + beta*EV[0])

Run the file to reproduce the solver comparisons of the class::

    python zurcher.py
"""

import numpy as np
import matplotlib.pyplot as plt


class zurcher():
    '''Harold Zurcher bus engine replacement model class, VFI version'''

    def __init__(self,
                 n = 175,           # number of state points
                 RC = 11.7257,      # replacement cost
                 c = 2.45569,       # parameter of maintance cost (theta_1)
                 p = [0.0937,0.4475,0.4459,0.0127],  # probabilities of transitions (theta_2)
                 beta = 0.9999):    # discount factor
        '''Init for the Zurcher model object'''
        self.RC, self.c, self.beta = RC, c, beta
        self.p = p   # transition probabilities, list; trpr is built by the n setter below
        self.n = n   # builds the grid and the transition matrix

    @property
    def n(self):
        '''Attribute getter for n'''
        return self.__n

    @n.setter
    def n(self, value):
        '''Attribute n setter'''
        if hasattr(self, '_zurcher__p'):   # p may not be set yet when n is assigned first
            assert len(self.__p) < value, 'More transition probability parameters than grid points'
        self.__n = value
        self.grid = np.arange(self.__n)
        if hasattr(self, '_zurcher__p'):
            self.trpr = self.__transition_probs()

    @property
    def p(self):
        '''Transition probabilities theta_2, all but the last (residual) one.
           Returns a copy: change p by assigning a new list, so the matrix is rebuilt'''
        return list(self.__p)

    @p.setter
    def p(self, value):
        '''Assigning p rebuilds the transition matrix'''
        value = [float(v) for v in value]
        assert min(value)>=0.0, 'Transition probability parameters must be non-negative'
        assert sum(value)<=1.0, 'Transition probability parameters must sum up to <1'
        self.__p = value
        if hasattr(self, '_zurcher__n'):   # n may not be set yet when p is assigned first
            assert len(value) < self.__n, 'More transition probability parameters than grid points'
            self.trpr = self.__transition_probs()

    def __repr__(self):
        '''String representation of the Zurcher model'''
        return 'Rust model of bus engine replacement (id={})'.format(id(self))

    def __transition_probs(self):
        '''Computing the transision probability matrix'''
        trpr = np.zeros((self.__n,self.__n))  # init
        probs = self.p + [1-sum(self.p)]  # ensure sum up to 1
        for i,p in enumerate(probs):
            trpr += np.diag([p]*(self.__n-i),k=i)
        trpr[:,-1] = 1.-np.sum(trpr[:,:-1],axis=1)  # last column absorbs the mass beyond the grid
        return trpr

    def bellman(self,ev0,deriv=False):
        '''Bellman operator for the model
           Depending on deriv argument, returns 2 or 3 outputs (Fréchet derivative)
        '''
        x = self.grid  # points in the next period state
        mcost = -0.001*x*self.c                         # 1-dim array of maintenance costs
        vx0 = mcost + self.beta * ev0                   # 1-dim array v(x,0), keep
        vx1 = mcost[0] - self.RC + self.beta * ev0[0]   # 1-dim array v(x,1), replace
        M = np.maximum(vx0,vx1)                         # de-max values to avoid exp(large number)
        logsum = M + np.log(np.exp(vx0-M) + np.exp(vx1-M))
        ev1 = self.trpr @ logsum                        # 1-dim array after matrix multiplication
        pk = 1/( np.exp(vx1-vx0)+1 )                    # choice prob to keep
        if not deriv:
            return ev1, pk
        # Fréchet derivative
        dev1 = self.beta * self.trpr * pk[np.newaxis,:] # element-wise, pk in rows
        dev1[:,0] += self.beta * self.trpr @ (1-pk)     # w.r.t. EV[0] special case
        return ev1, pk, dev1

    def _start(self, ev0):
        '''Starting point for the solvers: zeros unless one is given'''
        return np.zeros(self.n) if ev0 is None else np.asarray(ev0, dtype=float).copy()

    def solve_vfi(self, maxiter=100, tol=1e-6, callback=None, ev0=None):
        '''Solves the Rust model using value function iterations
        '''
        ev0 = self._start(ev0) # initial point for VFI
        err0 = 1.0 # initial lagged error
        for iter in range(maxiter):  # main loop
            ev1, pk = self.bellman(ev0)  # update approximation
            err = np.amax(np.abs(ev0-ev1))
            if callback:
                callback(iter=iter,model=self,ev1=ev1,ev0=ev0,err=err,err_prev=err0,pk=pk,method='vfi',itertype='sa')
            if err<tol:
                break  # break out if converged
            ev0 = ev1  # get ready to the next iteration
            err0 = err
        else:
            raise RuntimeError('Failed to converge in %d iterations'%maxiter)
        return ev1, pk

    def solve_nk(self, maxiter=100, tol=1e-6, callback=None, ev0=None):
        '''Solves the model using the Newton-Kantorovich iterations
        '''
        ev0 = self._start(ev0) # initial point
        err0 = 1.0 # initial lagged error
        for iter in range(maxiter):
            ev1,pk,dev = self.bellman(ev0,deriv=True) # compute with Fréchet derivative
            ev1 = ev0 - np.linalg.solve(np.eye(self.n)-dev,ev0 - ev1)  # NK step
            err = np.max(np.abs(ev1-ev0))
            if callback:
                callback(iter=iter,model=self,ev1=ev1,ev0=ev0,err=err,err_prev=err0,pk=pk,method='nk',itertype='nk')
            if err < tol:
                break  # break out if converged
            ev0 = ev1  # get ready to the next iteration
            err0 = err
        else:
            raise RuntimeError('Failed to converge in %d iterations'%maxiter)
        ev1,pk = self.bellman(ev1) # compute choice probabilities after convergence
        return ev1,pk

    def solve_poly(self,
                   maxiter=100,
                   tol=1e-10,
                   sa_min=5,         # minimum number of contraction steps
                   sa_max=25,        # maximum number of contraction steps
                   switch_tol=0.025, # tolerance of the switching rule
                   callback=None,
                   ev0=None):
        '''Solves the model using the poly-algorithm'''
        ev0 = self._start(ev0) # initial point
        err0 = 1.0 # initial lagged error
        nk = False # start with successive approximations
        for iter in range(maxiter):
            ev1,pk,dev = self.bellman(ev0,deriv=True) # update EV for both types of a step
            err = np.max(np.abs(ev1-ev0))
            nk = True if iter>= sa_max else nk  # have to switch to NK after sa_max
            nk = nk or (iter>=sa_min and abs(err/err0 - self.beta)<switch_tol)  # check if need to switch to NK
            if nk:
                ev1 = ev0 - np.linalg.solve(np.eye(self.n)-dev,ev0 - ev1)  # NK step
                err = np.max(np.abs(ev1-ev0))
            if callback:
                itertype = 'nk' if nk else 'sa'  # label for the iteration type
                callback(iter=iter,model=self,ev1=ev1,ev0=ev0,err=err,err_prev=err0,pk=pk,method='poly',itertype=itertype)
            if err < tol:
                break  # break out if converged
            ev0 = ev1  # get ready to the next iteration
            err0 = err
        else:
            raise RuntimeError('No convergence: maximum number of iterations achieved! Increase maxiter')
        ev1,pk = self.bellman(ev1) # compute choice probabilities after convergence
        return ev1,pk

    def solve_show(self,solver='vfi',verbosity=0,plot=True,**kvargs):
        '''Illustrate solution for given solver = {vfi,nk,poly} and
           print errors/relative errors from iterations (when verbose=True)
           All other arguments are passed to the solver
        '''
        if solver=='vfi':
            chosen_solver = self.solve_vfi
        elif solver=='nk':
            chosen_solver = self.solve_nk
        elif solver=='poly':
            chosen_solver = self.solve_poly
        else:
            raise RuntimeError('Unknown solver in solve_show()')
        if plot:
            fig1, (ax1,ax2) = plt.subplots(1,2,figsize=(14,8))
            ax1.grid(visible=True, which='both', color='0.65', linestyle='-')
            ax2.grid(visible=True, which='both', color='0.65', linestyle='-')
            ax1.set_xlabel('Mileage grid')
            ax2.set_xlabel('Mileage grid')
            ax1.set_title(f'Value function ({solver})')
            ax2.set_title(f'Probability of replacing the engine ({solver})')
        def callback(**argvars):
            iter,itertype,err,derr = argvars['iter'],argvars['itertype'],argvars['err'],argvars['err_prev']
            mod, ev, pk = argvars['model'],argvars['ev1'],argvars['pk']
            if verbosity>1:
                if iter==0:
                    print('Solver = %s'%solver)
                    print('-'*42)
                    print('%7s %16s %16s'%('iter','err','err(i)/err(i-1)'))
                    print('-'*42)
                print('%4d %2s %16.4e %16.12f'%(iter,itertype[:2],err,err/derr))
            elif verbosity>0:
                if iter==0:
                    print('Solver = %s'%solver)
                    print('-'*22)
                    print('%4s %16s'%('iter','err'))
                    print('-'*22)
                print('%4d %16.4e'%(iter,err))
            if plot:
                ax1.plot(mod.grid,ev,color='k',alpha=0.25)
                ax2.plot(mod.grid,pk,color='k',alpha=0.25)
            callback.nriter = iter+1  # number of iterations run, saved in function object attribute
        # run the chosen solver
        ev,pk = chosen_solver(callback=callback,**kvargs)
        if plot:
            # add solutions
            ax1.plot(self.grid,ev,color='r',linewidth=2.5)
            ax2.plot(self.grid,pk,color='r',linewidth=2.5)
            plt.show()
        print('{} solved with {} in {} iterations'.format(self,solver,callback.nriter))
        return ev,pk

# This runs when the module is executed as a script, not when imported
if __name__ == '__main__':
    import timeit

    # compare SA, NK at a low discount factor
    model = zurcher(beta=0.9)
    ev1,pk1 = model.solve_show(maxiter=1500)
    ev2,pk2 = model.solve_show(solver='nk')
    print('Max diff between value functions is ' ,np.amax(np.abs(ev1-ev2)))
    print('Max diff between policy functions is',np.amax(np.abs(pk1-pk2)))
    print()

    # convergence rate of VFI is beta: raise it and count iterations
    model = zurcher(beta=0.975)
    ev1,pk1 = model.solve_show(maxiter=1500,verbosity=1,plot=False)
    ev2,pk2 = model.solve_show(solver='nk',verbosity=1,plot=False)
    print('Max diff between value functions is ' ,np.amax(np.abs(ev1-ev2)))
    print('Max diff between policy functions is',np.amax(np.abs(pk1-pk2)))
    print()

    # when to switch from SA to NK: watch the error ratio approach beta
    model = zurcher(beta=0.975)
    model.solve_show(maxiter=1500,verbosity=2,plot=False)
    print()

    # SA, NK and the poly-algorithm on the same problem
    m = zurcher(beta=0.975)
    ev,pk = m.solve_show(tol=1e-10,maxiter=1500)
    ev,pk = m.solve_show(tol=1e-10,solver='nk',plot=False)
    polyset = {'sa_min':10,
               'sa_max':100,
               'switch_tol':0.000215,
              }
    ev,pk = m.solve_show(tol=1e-10,verbosity=2,solver='poly',**polyset)
    print()

    # original parameters from Rust 1987: pure VFI is hopeless, poly takes milliseconds
    m = zurcher()
    polyset = {'sa_min':10,
               'sa_max':100,
               'switch_tol':0.0005,
              }
    ev,pk = m.solve_show(tol=1e-10,solver='poly',**polyset)
    reps = 50
    t = timeit.timeit(lambda: m.solve_poly(tol=1e-10,**polyset), number=reps)
    print(f'poly-algorithm at Rust parameters: {1000*t/reps:.2f} ms per solve')
