from optuna_fabrics.tune.panda_trial import PandaTrial
from optuna_fabrics.tune.table_trial import TableTrial

class PandaTableTrial(TableTrial, PandaTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._sub_goal_0_links = ['panda_link0', 'panda_hand']
        self._sub_goal_1_links = ['panda_link7', 'panda_hand']

