import numpy as np
import pandas as pd

from CommonFiles.pBase import pBase
from CommonFiles.Futures import map_

from CommonFiles.Models.degModelFinal import create_class
from CommonFiles.Models.DegModelStoch import simulate_stoch

vol = 225
periods = 10
ntraj = 10000
group_size = 1000
num_groups = ntraj/group_size


base_control = create_class()

def param_reduction(param, amount, base_list):
    t_param = np.array(base_control.paramset)
    t_param[base_control.pdict[param]] *= (1-amount)
    tbase = pBase(base_control.model, t_param, base_list[-1].y0)
    try:
        tbase.approxY0(tout=1000, tol=1E-5)
        tbase.solveBVP()
        assert tbase.findstationary() == 0
        base_list += [tbase]
    except Exception: pass
        # base_list += [np.nan]

vac1p_base_list = [base_control]
vac1p_vals = np.linspace(0, 0.95, num=100)
for i in vac1p_vals[1:]:
    param_reduction('vaC1P', i, vac1p_base_list)

vdcn_base_list = [base_control]
vdcn_vals = np.linspace(0, 0.95, num=100)
for i in vdcn_vals[1:]:
    param_reduction('vdCn', i, vdcn_base_list)


def estimate_decay(base, job_id):
    ts, traj = simulate_stoch(base, vol, t=periods*base_control.y0[-1],
                              traj=ntraj,
                              increment=base_control.y0[-1]/100,
                              job_id=job_id)

    # Split traj into equal parts to estimate grand mean and variance
    from CommonFiles.StochDecayEstimator import StochDecayEstimator
    trans = 0

    fit_dict = {}
    for i in xrange(num_groups):
        traj_i = traj[(i*group_size):((i+1)*group_size)]

        master = StochDecayEstimator(ts[trans:],
                                     traj_i.mean(0)[trans:,:], base)
        fit_dict.update({
            # 'A'+str(i) : master.sinusoid_params['amplitude'],
            'T'+str(i) : master.sinusoid_params['period'],
            'DS'+str(i) : master.sinusoid_params['decay'],
            'D'+str(i) : master.decay,
            'B'+str(i) : base.y0[-1],
            'R'+str(i) : master.r2,
        })

    return pd.Series(fit_dict)

def estimate_decay_vac1p(ind):
    return estimate_decay(vac1p_base_list[ind], job_id='vac1p' + str(ind))

def estimate_decay_vdcn(ind):
    return estimate_decay(vdcn_base_list[ind], job_id='vdcn' + str(ind))

if __name__ == "__main__":

    inds_vac1p = np.linspace(0, len(vac1p_base_list)-1, num=20,
                             endpoint=True).astype(int)
    inds_vdcn  = np.linspace(0, len(vdcn_base_list)-1, num=20,
                             endpoint=True).astype(int)

    vac1p_results = list(map_(estimate_decay_vac1p, inds_vac1p))
    vac1p_results_df = pd.concat(vac1p_results, axis=1).T
    vac1p_results_df.to_pickle('vac1p_results.p')
    
    vdcn_results = list(map_(estimate_decay_vdcn, inds_vdcn))
    vdcn_results_df = pd.concat(vdcn_results, axis=1).T
    vdcn_results_df.to_pickle('vdcn_results.p')

