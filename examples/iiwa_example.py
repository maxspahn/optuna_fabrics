from optuna_fabrics.tune.iiwa_ring_trial import IiwaRingTrial
from optuna_fabrics.tune.iiwa_table_trial import IiwaTableTrial
from optuna_fabrics.tune.fabrics_study import FabricsStudy

if __name__ == "__main__":
    weights_objective = {'path_length': 0.0, 'time_to_goal': 1.0, 'obstacles': 0.0}
    iiwa_trial = IiwaRingTrial(weights=weights_objective)
    #iiwa_trial = IiwaTableTrial(weights=weights_objective)
    study = FabricsStudy(iiwa_trial)
    study.run()
    #study.show_history()
