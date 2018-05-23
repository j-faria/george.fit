import numpy as np
import scipy.optimize as op

def optimization(gp, t, y, **minimize_kwargs):
    # Define the objective function (negative log-likelihood in this case).
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    # And the gradient of the objective function.
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)

    # You need to compute the GP once before starting the optimization.
    gp.compute(t)

    # Print the initial ln-likelihood.
    print('Initial log-likelihood:', gp.log_likelihood(y))

    # Run the optimization routine.
    p0 = gp.get_parameter_vector()
    
    minimize_kwargs.setdefault('jac', grad_nll) # provide our own
    minimize_kwargs.setdefault('method', "L-BFGS-B")

    results = op.minimize(nll, p0, **minimize_kwargs)

    # Update the kernel and print the final log-likelihood.
    gp.set_parameter_vector(results.x)
    print('Final log-likelihood:', gp.log_likelihood(y))

    return results