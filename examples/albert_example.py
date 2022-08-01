from optuna_fabrics.tune.albert_ring_trial import AlbertRingTrial
#from optuna_fabrics.tune.albert_reach_trial import AlbertReachTrial
#from optuna_fabrics.tune.albert_table_trial import AlbertTableTrial
from optuna_fabrics.tune.fabrics_study import FabricsStudy

if __name__ == "__main__":
    weights_objective = {'path_length': 0.0, 'time_to_goal': 1.0, 'obstacles': 0.0}
    #albert_trial = AlbertTableTrial(weights=weights_objective)
    albert_trial = AlbertRingTrial(weights=weights_objective)
    #albert_trial = AlbertReachTrial(weights=weights_objective)
    study = FabricsStudy(albert_trial)
    study.run()
    #study.show_history()
