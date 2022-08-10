from optuna_fabrics.tune.panda_ring_trial import PandaRingTrial
from optuna_fabrics.tune.panda_reach_trial import PandaReachTrial
from optuna_fabrics.tune.panda_table_trial import PandaTableTrial
from optuna_fabrics.tune.fabrics_study import FabricsStudy

if __name__ == "__main__":
    weights_objective = {'path_length': 0.1, 'time_to_goal': 0.7, 'obstacles': 0.2}
    panda_trial = PandaTableTrial(weights=weights_objective)
    #panda_trial = PandaRingTrial(weights=weights_objective)
    #panda_trial = PandaReachTrial(weights=weights_objective)
    study = FabricsStudy(panda_trial)
    study.run()
    #study.show_history()
