import numpy as np
import pandas as pd

# from CommonFiles.pBase import pBase
from CommonFiles.Futures import map_

from CommonFiles.Models.degModelFinal import create_class
from CommonFiles.Models.DegModelStoch import simulate_stoch

base = create_class()
# vol = 1000
periods = 10
ntraj = 10000 
group_size = 1000
num_groups = ntraj/group_size

def test_estimate_decay(vol):
    return np.random.rand(num_groups)

def estimate_decay(vol):
    ts, traj = simulate_stoch(base, vol, t=periods*base.y0[-1],
                              traj=ntraj, increment=base.y0[-1]/100,
                              job_id=vol)

    # Split traj into equal parts to estimate grand mean and variance
    from CommonFiles.StochDecayEstimator import StochDecayEstimator
    trans = 0

    fit_dict = {'V' : vol}
    for i in xrange(num_groups):
        traj_i = traj[(i*group_size):((i+1)*group_size)]

        master = StochDecayEstimator(ts[trans:],
                                     traj_i.mean(0)[trans:,:], base)

        fit_dict.update({
            'T'+str(i) : master.sinusoid_params['period'],
            'DS'+str(i) : master.sinusoid_params['decay'],
            'D'+str(i) : master.decay,
            'B'+str(i) : base.y0[-1],
            'R'+str(i) : master.r2,
        })

    return pd.Series(fit_dict)

if __name__ == "__main__":
    vols = np.logspace(2, 2.7, num=25, endpoint=True).astype(int)
    results = list(map_(estimate_decay, vols))
    results_df = pd.concat(results, axis=1).T
    results_df.to_pickle('volume_results.p')
