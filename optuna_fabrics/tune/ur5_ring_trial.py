from optuna_fabrics.tune.ur5_trial import Ur5Trial
from optuna_fabrics.tune.ring_trial import RingTrial

class Ur5RingTrial(RingTrial, Ur5Trial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._sub_goal_0_links = ['base_link', 'ee_link']
        self._sub_goal_1_links = ['wrist_3_link', 'ee_link']

