
import sys
sys.path.append('../../george/build/lib.linux-x86_64-3.6')

import george
from george.modeling import Model
from george import kernels
import numpy as np
import matplotlib.pyplot as pl



"""
Simulated dataset from 
http://george.readthedocs.io/en/latest/tutorials/model
"""
class Model(Model):
    parameter_names = ("amp", "location", "log_sigma2")

    def get_value(self, t):
        return self.amp * \
               np.exp(-0.5*(t.flatten() - self.location)**2 * \
                        np.exp(-self.log_sigma2)
                     )

np.random.seed(1234)

def generate_data(params, N, rng=(-5, 5)):
    gp = george.GP(0.1 * kernels.ExpSquaredKernel(3.3))
    t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))
    y = gp.sample(t)
    y += Model(**params).get_value(t)
    yerr = 0.05 + 0.05 * np.random.rand(N)
    y += yerr * np.random.randn(N)
    return t, y, yerr

truth = dict(amp=-1.0, location=0.1, log_sigma2=np.log(0.4))
t, y, yerr = generate_data(truth, 50)
np.savetxt('simulated_dataset.dat', np.vstack([t,y,yerr]).T)


pl.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
pl.ylabel(r"$y$")
pl.xlabel(r"$t$")
pl.xlim(-5, 5)
pl.title("simulated data")
pl.savefig('simulated_dataset.png')





"""
Data in Figure 5.6 in Chapter 5 of Rasmussen & Williams (2006). 
The data are measurements of the atmospheric CO2 concentration 
made at Mauna Loa, Hawaii (Keeling & Whorf 2004). 
The dataset is available from the statsmodels package
http://www.statsmodels.org/devel/datasets/generated/co2.html
or from the source at 
http://cdiac.ess-dive.lbl.gov/trends/co2/sio-keel-flask/sio-keel-flaskmlo_c.html
"""
from statsmodels.datasets import co2

data = co2.load_pandas().data
t = 2000 + (np.array(data.index.to_julian_date()) - 2451545.0) / 365.25
y = np.array(data.co2)
m = np.isfinite(t) & np.isfinite(y) & (t < 1996)
t, y = t[m][::4], y[m][::4]

np.savetxt('MaunaLoa_dataset.dat', np.vstack([t,y]).T)

pl.plot(t, y, ".k")
pl.xlim(t.min(), t.max())
pl.ylim(310, 370)
pl.xlabel("year")
pl.ylabel("CO$_2$ in ppm")
pl.savefig('MaunaLoa_dataset.png')