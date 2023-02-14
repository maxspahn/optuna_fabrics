from optuna_fabrics.tune.ur5_ring_trial import Ur5RingTrial
from optuna_fabrics.tune.fabrics_study import FabricsStudy

if __name__ == "__main__":
    weights_objective = {'path_length': 0.1, 'time_to_goal': 0.7, 'obstacles': 0.2}
    ur5_trial = Ur5RingTrial(weights=weights_objective)
    #ur5_trial = Ur5TableTrial(weights=weights_objective)
    study = FabricsStudy(ur5_trial)
    study.run()
    #study.show_history()
