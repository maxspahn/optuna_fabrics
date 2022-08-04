from typing import Dict
import os

import numpy as np
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
from fabrics.planner.serialized_planner import SerializedFabricPlanner

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
 


class PandaTrial(FabricsTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 7
        self._q0 = np.array([0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
        self._qdot0 = np.zeros(7)
        self._collision_links = ['panda_link8', 'panda_link4', "panda_link7", "panda_link5", "panda_hand"]
        self._link_sizes = {
            'panda_link8': 0.1,
            'panda_link4': 0.1,
            "panda_link7": 0.08,
            "panda_link5": 0.08,
            "panda_hand": 0.08
        }
        self._ee_link = "panda_hand"
        self._root_link = "panda_link0"
        self._self_collision_pairs = {
            "panda_hand": ['panda_link2', 'panda_link4'], 
        }
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        self._urdf_file = self._absolute_path + "/panda.urdf"
        with open(self._urdf_file, "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, 'panda_link0', ['panda_hand', 'panda_link5_offset'])


    def set_planner(self):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        serialize_file = self._absolute_path + "/../planner/serialized_planners/panda_planner.pkl"
        if os.path.exists(serialize_file):
            planner = SerializedFabricPlanner(serialize_file)
            return planner
        robot_type = "panda"
        planner = SymbolicFabricPlanner(
            self._degrees_of_freedom,
            robot_type,
            urdf=self._urdf,
            root_link=self._root_link,
            end_link=[self._ee_link],
        )
        panda_limits = [
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8974, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973]
            ]
        # The planner hides all the logic behind the function set_components.
        goal = self.dummy_goal()
        planner.set_components(
            self._collision_links,
            self._self_collision_pairs,
            goal,
            number_obstacles=self._number_obstacles,
            limits=panda_limits,
        )
        planner.concretize()
        planner.serialize(serialize_file)
        return planner







