from typing import Dict
import os
import casadi as ca
import gym
import planarenvs.point_robot


import numpy as np
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
from fabrics.planner.serialized_planner import SerializedFabricPlanner

from forwardkinematics.planarFks.pointFk import PointFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
 


class PointTrial(FabricsTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 2
        self._q0 = np.array([-3.0, 1.0])
        self._qdot0 = np.zeros(2)
        self._collision_links = [1]
        self._link_sizes = {
            1: 1.0,
        }
        self._ee_link = 1
        self._root_link = 0
        self._self_collision_pairs = {}
        self._fk = PointFk()
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))

    def initialize_environment(self, render: bool=True):
        env = gym.make("point-robot-acc-v0", dt=0.01, render=render)
        return env

    def create_collision_metric(self, obstacles):
        q = ca.SX.sym("q", self._degrees_of_freedom)
        distance_to_obstacles = 10000
        for link in self._collision_links:
            fk = self._fk.fk(q, link, positionOnly=True)
            for obst in obstacles:
                obst_position = np.array(obst.position())
                distance_to_obstacles = ca.fmin(distance_to_obstacles, ca.norm_2(obst_position - fk))
        self._collision_metric = ca.Function("collision_metric", [q], [distance_to_obstacles])

    def set_planner(self):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        serialize_file = self._absolute_path + "/../planner/serialized_planners/point_planner.pkl"
        if os.path.exists(serialize_file):
            planner = SerializedFabricPlanner(serialize_file)
            return planner
        robot_type = "pointRobot"
        planner = SymbolicFabricPlanner(
            self._degrees_of_freedom,
            robot_type,
            root_link=self._root_link,
            end_link=[self._ee_link],
        )
        panda_limits = [
                [-5, 5],
                [-5, 5],
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







