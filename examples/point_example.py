from optuna_fabrics.tune.point_trial import PointTrial
from optuna_fabrics.tune.fabrics_study import FabricsStudy

if __name__ == "__main__":
    point_trial = PointTrial()
    study = FabricsStudy(point_trial)
    study.run()
