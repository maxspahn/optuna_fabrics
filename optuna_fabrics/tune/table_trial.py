import gym
from typing import Dict, Any
import logging
import optuna
import os
import warnings

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

import numpy as np
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
from fabrics.planner.serialized_planner import SerializedFabricPlanner

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
import quaternionic

logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)

def generate_random_orientation(mean, rotation=0.0, tilting=0.0):
    """
    Generate random orientation of feasible reaching tasks.

    params
    mean:list
        mean of quaternion, good: mean = [0.0, 0.707, 0.0, 0.0]
    rotation: float
        amount of rotation around z axis
    tilting: float
        amount of tilting
    """
    lower_limit = np.array([0, 0, -rotation, -tilting])
    upper_limit = np.array([0, 0, rotation, tilting])
    orientation = np.array(mean) + np.random.uniform(lower_limit, upper_limit, 4)
    return orientation.tolist()


class TableTrial(FabricsTrial):
    def __init__(self, weights=None):
        self._number_obstacles = 10
        super().__init__(weights)

    def dummy_goal(self):
        goal_dict = {
            "subgoal0": {
                "weight": 0.0,
                "is_primary_goal": True,
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
                "weight": 3.0,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": self._sub_goal_1_links[0],
                "child_link": self._sub_goal_1_links[1],
                "desired_position": [0.107, 0, 0],
                "angle": [1.0, 0.0, 0.0, 0.0],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        return GoalComposition(name="goal", content_dict=goal_dict)

    def shuffle_env(self, env, shuffle=True):
        # Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": self._sub_goal_0_links[0],
                "child_link": self._sub_goal_0_links[1],
                "desired_position": [0.7, -0.2, 0.07],
                "epsilon": 0.05,
                "low": [0.4, -0.5, 0.07],
                "high": [0.8, 0.5, 0.07],
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 3.0,
                "is_primary_goal": False,
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
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        if shuffle:
            goal.shuffle()
        env.add_goal(goal)
        # Definition of the obstacle.
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [1000.0, 0.5, 0.1], "radius": 0.1},
            "low": {'position': [0.2, -0.7, 0.0], 'radius': 0.05},
            "high": {'position': [1.0, 0.7, 0.2], 'radius': 0.15},
        }
        static_obstacles_positions = [
            [0.4201, 0.3462, 0.03198],
            [0.8695, -0.585, 0.06963],
            [0.7850, 0.3850, 0.08218],
            [0.4572, 0.1572, 0.07123],
            [0.4940, -0.494, 0.02137],
            [0.3449, 0.0449, 0.1479],
            [0.3172, 0.3172, 0.1815],
            [0.8994, -0.099, 0.0556],
            [0.8231, 0.8231, 0.03824],
            [0.7101, -0.710, 0.01567],
            [0.2948, 0.2148, 0.04941],
        ]
        obstacles = []
        for i in range(self._number_obstacles):
            obst_i = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
            if shuffle:
                obst_i.shuffle()
                while np.linalg.norm(
                        np.array(obst_i.position()[0:2])
                        - np.array(goal.primary_goal().position()[0:2])
                    ) < 0.15:
                    obst_i.shuffle()
            else:
                obst_i._config.geometry.position = static_obstacles_positions[i]
            obstacles.append(obst_i)

        for obst in obstacles:
            env.add_obstacle(obst)
        return env, obstacles, goal

    def evaluate_distance_to_goal(self, q: np.ndarray):
        sub_goal_0_position = np.array(self._goal.sub_goals()[0].position())
        fk = self._generic_fk.fk(q, self._goal.sub_goals()[0].parent_link(), self._goal.sub_goals()[0].child_link(), positionOnly=True)
        return np.linalg.norm(sub_goal_0_position - fk) / self._initial_distance_to_goal_0 


    def set_goal_arguments(self, q0: np.ndarray, goal:GoalComposition, arguments):
        self._goal = goal
        sub_goal_0_position = np.array(goal.sub_goals()[0].position())
        sub_goal_1_position = np.array(goal.sub_goals()[1].position())
        sub_goal_1_quaternion = quaternionic.array(goal.sub_goals()[1].angle())
        sub_goal_1_rotation_matrix = sub_goal_1_quaternion.to_rotation_matrix
        fk_0 = self._generic_fk.fk(q0, goal.sub_goals()[0].parent_link(), goal.sub_goals()[0].child_link(), positionOnly=True)
        self._initial_distance_to_goal_0 = np.linalg.norm(sub_goal_0_position - fk_0)
        #self._initial_distance_to_goal_0 = 1.0
        arguments['x_goal_0'] = sub_goal_0_position
        arguments['x_goal_1'] = sub_goal_1_position
        arguments['angle_goal_1'] = sub_goal_1_rotation_matrix
        arguments['weight_goal_0']=np.array([1.0])
        arguments['weight_goal_1']=np.array([2.0])
        return self._initial_distance_to_goal_0

