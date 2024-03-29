import gym
from typing import Dict, Any
import logging
import optuna
import os
import warnings
from abc import abstractmethod

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

import numpy as np
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
from fabrics.planner.serialized_planner import SerializedFabricPlanner


from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
import quaternionic
 


class IiwaTrial(FabricsTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 7
        self._q0 = np.array([0, -0.6, 0.0, -1.1, 0.00, 0, 0.0])
        self._qdot0 = np.zeros(self._degrees_of_freedom)
        self._collision_links = ["iiwa_link_3", "iiwa_link_5", "iiwa_link_7", "iiwa_link_ee"]
        self._link_sizes = {
            'iiwa_link_3': 0.1,
            'iiwa_link_5': 0.1,
            "iiwa_link_7": 0.08,
            "iiwa_link_ee": 0.08,
        }
        self._self_collision_pairs = {
            "iiwa_link_7": ['iiwa_link_3', 'iiwa_link_5']
        }
        self._ee_link = "iiwa_link_ee"
        self._root_link = "iiwa_link_0"
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        self._urdf_file = self._absolute_path + "/iiwa7.urdf"
        with open(self._urdf_file, "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, self._root_link, self._ee_link)

    def set_planner(self):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        serialize_file = self._absolute_path + "/../planner/serialized_planners/iiwa_planner.pkl"
        if os.path.exists(serialize_file):
            planner = SerializedFabricPlanner(serialize_file)
            return planner
        robot_type = "panda"

        ## Optional reconfiguration of the planner
        # base_inertia = 0.03
        # attractor_potential = "20 * ca.norm_2(x)**4"
        # damper = {
        #     "alpha_b": 0.5,
        #     "alpha_eta": 0.5,
        #     "alpha_shift": 0.5,
        #     "beta_distant": 0.01,
        #     "beta_close": 6.5,
        #     "radius_shift": 0.1,
        # }
        # planner = ParameterizedFabricPlanner(
        #     degrees_of_freedom,
        #     robot_type,
        #     base_inertia=base_inertia,
        #     attractor_potential=attractor_potential,
        #     damper=damper,
        # )
        planner = SymbolicFabricPlanner(
            self._degrees_of_freedom,
            robot_type,
            urdf=self._urdf,
            root_link='iiwa_link_0',
            end_link=['iiwa_link_ee'],
        )
        iiwa_limits= [
            [-2.96705973, 2.96705973],
            [-2.0943951, 2.0943951],
            [-2.96705973, 2.96705973],
            [-2.0943951, 2.0943951],
            [-2.96705973, 2.96705973],
            [-2.0943951, 2.0943951],
            [-3.05432619, 3.05432619],
        ]

        # The planner hides all the logic behind the function set_components.
        goal = self.dummy_goal()
        planner.set_components(
            self._collision_links,
            self._self_collision_pairs,
            goal,
            number_obstacles=self._number_obstacles,
            limits=iiwa_limits,
        )
        planner.concretize()
        planner.serialize(serialize_file)
        return planner

