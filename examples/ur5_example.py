from optuna_fabrics.tune.ur5_ring_trial import Ur5RingTrial
from optuna_fabrics.tune.fabrics_study import FabricsStudy

if __name__ == "__main__":
    weights_objective = {'path_length': 0.0, 'time_to_goal': 1.0, 'obstacles': 0.0}
    ur5_trial = Ur5RingTrial(weights=weights_objective)
    #ur5_trial = Ur5TableTrial(weights=weights_objective)
    study = FabricsStudy(ur5_trial)
    study.run()
    #study.show_history()
