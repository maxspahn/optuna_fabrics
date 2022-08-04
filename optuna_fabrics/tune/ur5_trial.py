import gym
from typing import Dict, Any
import logging
import optuna
import os
import warnings
from abc import abstractmethod

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle

import numpy as np
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
from fabrics.planner.serialized_planner import SerializedFabricPlanner

import urdfenvs.generic_urdf_reacher

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
import quaternionic
 


class Ur5Trial(FabricsTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 6
        self._q0 = np.array([3.14, -0.7, -1.57, -0.8, 1.57, 0])
        self._qdot0 = np.zeros(self._degrees_of_freedom)
        self._collision_links = ['shoulder_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'forearm_link']
        self._link_sizes = {
            'shoulder_link': 0.1,
            'wrist_1_link': 0.05,
            "wrist_2_link": 0.05,
            "wrist_3_link": 0.05,
            "forearm_link": 0.08,
        }
        self._self_collision_pairs = {
            "wrist_3_link": ['forearm_link', 'shoulder_link']
        }
        self._ee_link = "ee_link"
        self._root_link = "base_link"
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        self._urdf_file = self._absolute_path + "/ur5.urdf"
        with open(self._urdf_file, "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, 'base_link', 'wrist_3_link')



    def set_planner(self):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        serialize_file = self._absolute_path + "/../planner/serialized_planners/ur5_planner.pkl"
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
            root_link=self._root_link,
            end_link=self._ee_link,
        )
        ur5_limits= [
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-1 * np.pi, 1 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
            ]
        # The planner hides all the logic behind the function set_components.
        goal = self.dummy_goal()
        planner.set_components(
            self._collision_links,
            self._self_collision_pairs,
            goal,
            number_obstacles=self._number_obstacles,
            limits=ur5_limits,
        )
        planner.concretize()
        planner.serialize(serialize_file)
        return planner



