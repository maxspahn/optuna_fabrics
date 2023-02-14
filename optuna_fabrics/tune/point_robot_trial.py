from optuna_fabrics.tune.point_avoidance_trial import PointAvoidanceTrial
from optuna_fabrics.tune.point_trial_2 import PointTrial

class PointRobotTrial(PointAvoidanceTrial, PointTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._sub_goal_0_links = [0, 1]

