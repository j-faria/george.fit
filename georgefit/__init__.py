import sys
sys.path.append('../george/build/lib.linux-x86_64-3.6')

import george
print('Usinge george v' + george.__version__)

from .optimization import optimization
from .mcmc import mcmc