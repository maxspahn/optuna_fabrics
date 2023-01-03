from optuna_fabrics.tune.point_robot_trial import PointRobotTrial
from optuna_fabrics.tune.fabrics_study import FabricsStudy

if __name__ == "__main__":
    point_trial = PointRobotTrial()
    study = FabricsStudy(point_trial)
    study.run()
