fodeint
======
| Numerical integration of fractional ordinary differential equations.

Overview
--------
fodeint will be a collection of numerical algorithms for integrating fractional ordinary differential equations (FODEs). It has simple functions that can be used in a similar way to ``scipy.integrate.odeint()`` or MATLAB's ``ode45``.

This first version supports Caputo equations with fractional exponent in the range (0,1).

This is prototype code in python, so not aiming for speed. Later can always rewrite these with loops in C when speed is needed.

Warning: this is an early pre-release. Wait for version 1.0. Bug reports are very welcome!

functions
---------
| ``fodeint(a, f, y0, tspan)`` for Caputo equation D^a y(t) = f(y,t), 0<a<1.

This works with scalar or vector equations.

specific algorithms:
--------------------
| ``caputoEuler(a, f, y0, tspan)``: an explicit one-step Adams-Bashforth (Euler) method. The simplest possible method.

Examples:
---------

References for these algorithms:
--------------------------------

| ``caputoEuler``: 
| K. Diethelm, N. J. Ford and A. D. Freed (2004) Detailed error analysis for a fractional Adams method 
| C. Li and F. Zeng (2012) Finite Difference Methods for Fractional Differential Equations

TODO
----
- Implement fast sum-of-exponentials approximation of Jiang et al. (2017).
- Support other values of the fractional exponent.
- Implement a higher order predictor-corrector algorithm.
- Support Caputo equations with noise (FSODEs).
- Support Riemann-Liouville and Riesz equations. 
