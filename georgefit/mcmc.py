import numpy as np
import emcee

def mcmc(gp, t, y, burn=200, sample=200, nwalkers=36):
    def lnprob(p):
        # Trivial uniform prior.
        if np.any((-100 > p[1:]) + (p[1:] > 100)):
            return -np.inf

        # Update the kernel and compute the lnlikelihood.
        gp.set_parameter_vector(p)
        return gp.lnlikelihood(y, quiet=True)


    gp.compute(t)

    # Set up the sampler.
    ndim = len(gp)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Initialize the walkers.
    p0 = gp.get_parameter_vector() \
            + 1e-4 * np.random.randn(nwalkers, ndim)

    print("Running burn-in (%d steps)" % burn)
    p0, _, _ = sampler.run_mcmc(p0, burn)

    print("Running production chain (%d steps)" % sample)
    sampler.run_mcmc(p0, sample)

    return sampler