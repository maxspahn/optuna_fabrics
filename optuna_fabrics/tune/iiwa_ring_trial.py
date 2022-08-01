from optuna_fabrics.tune.iiwa_trial import IiwaTrial
from optuna_fabrics.tune.ring_trial import RingTrial

class IiwaRingTrial(RingTrial, IiwaTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._sub_goal_0_links = ['iiwa_link_0', 'iiwa_link_ee']
        self._sub_goal_1_links = ['iiwa_link_7', 'iiwa_link_ee']

