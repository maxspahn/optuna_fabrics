import gym
from typing import Dict, Any
import logging
import optuna
import os
import warnings

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle

import numpy as np
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
from fabrics.planner.serialized_planner import SerializedFabricPlanner

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
import quaternionic

logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)

class TableTrial(FabricsTrial):
    def __init__(self, weights=None):
        self._number_obstacles = 5
        super().__init__(weights)

    def dummy_goal(self):
        goal_dict = {
            "subgoal0": {
                "m": 3,
                "w": 0.0,
                "prime": True,
                "indices": [0, 1, 2],
                "parent_link": self._sub_goal_0_links[0],
                "child_link": self._sub_goal_0_links[1],
                "desired_position": [0.7, 0.1, 0.8],
                "high": [0.7, 0.2, 0.7],
                "low": [0.5, -0.2, 0.6],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "m": 3,
                "w": 3.0,
                "prime": False,
                "indices": [0, 1, 2],
                "parent_link": self._sub_goal_1_links[0],
                "child_link": self._sub_goal_1_links[1],
                "desired_position": [0.107, 0, 0],
                "angle": [1.0, 0.0, 0.0, 0.0],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        return GoalComposition(name="goal", contentDict=goal_dict)


    def shuffle_env(self, env, shuffle=True):
        # Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "m": 3,
                "w": 1.0,
                "prime": True,
                "indices": [0, 1, 2],
                "parent_link": self._sub_goal_0_links[0],
                "child_link": self._sub_goal_0_links[1],
                "desired_position": [0.5, 0.0, 0.1],
                "epsilon": 0.05,
                "low": [0.4, -0.5, 0.05],
                "high": [0.8, 0.5, 0.05],
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "m": 3,
                "w": 3.0,
                "prime": False,
                "indices": [0, 1, 2],
                "parent_link": self._sub_goal_1_links[0],
                "child_link": self._sub_goal_1_links[1],
                "desired_position": [0.0, 0.0, 0.107],
                #"angle": [0, -np.sqrt(2), 0.0, np.sqrt(2)], # facing downwords
                "angle": [0, 1, 0, 0],
                "low": [0.0, 0.0, 0.107],
                "high": [0.0, 0.0, 0.107],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        goal = GoalComposition(name="goal", contentDict=goal_dict)
        if shuffle:
            goal.shuffle()
        env.add_goal(goal)
        # Definition of the obstacle.
        static_obst_dict = {
            "dim": 3,
            "type": "sphere",
            "geometry": {"position": [1000.0, 0.5, 0.1], "radius": 0.1},
            "low": {'position': [0.2, -0.7, 0.0], 'radius': 0.05},
            "high": {'position': [1.0, 0.7, 0.4], 'radius': 0.1},
        }
        obstacles = []
        for i in range(self._number_obstacles):
            obst_i = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
            if shuffle:
                obst_i.shuffle()
            obstacles.append(obst_i)
        return env, obstacles, goal

    def evaluate_distance_to_goal(self, q: np.ndarray):
        sub_goal_0_position = np.array(self._goal.subGoals()[0].position())
        fk = self._generic_fk.fk(q, self._goal.subGoals()[0].parentLink(), self._goal.subGoals()[0].childLink(), positionOnly=True)
        return np.linalg.norm(sub_goal_0_position - fk) / self._initial_distance_to_goal_0 

    def set_goal_arguments(self, q0: np.ndarray, goal:GoalComposition):
        self._goal = goal
        arguments = {}
        sub_goal_0_position = np.array(goal.subGoals()[0].position())
        sub_goal_1_position = np.array(goal.subGoals()[1].position())
        sub_goal_1_quaternion = quaternionic.array(goal.subGoals()[1].angle())
        sub_goal_1_rotation_matrix = sub_goal_1_quaternion.to_rotation_matrix
        fk_0 = self._generic_fk.fk(q0, goal.subGoals()[0].parentLink(), goal.subGoals()[0].childLink(), positionOnly=True)
        self._initial_distance_to_goal_0 = np.linalg.norm(sub_goal_0_position - fk_0)
        arguments['x_goal_0'] = sub_goal_0_position
        arguments['x_goal_1'] = sub_goal_1_position
        arguments['angle_goal_1'] = sub_goal_1_rotation_matrix
        return arguments, self._initial_distance_to_goal_0

