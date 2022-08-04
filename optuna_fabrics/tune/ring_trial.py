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

def generate_random_position():
    return np.random.uniform(np.array([0.5, -0.2, 0.6]), np.array([0.7, 0.2, 0.7]), 3)


class RingTrial(FabricsTrial):
    def __init__(self, weights=None):
        self._number_obstacles = 10
        self._obstacle_resolution = self._number_obstacles
        super().__init__(weights=weights)

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


    def shuffle_env(self, env, randomize=True):
        # Definition of the goal.
        mean = [0.0, 0.707, 0.0, 0.0]
        if randomize:
            goal_orientation = generate_random_orientation(mean, rotation=0.1, tilting=0.1)
            goal_position = generate_random_position().tolist()
            ring_orientation = generate_random_orientation(goal_orientation, rotation=0.1, tilting=0.1)

        else:
            goal_orientation = [0.3, 0.707, 0.15, 0.0]
            goal_position = [0.59, 0.15, 0.76]
            ring_orientation = [-0.10, 0.71, -0.20, 0.0]
        goal_dict = {
            "subgoal0": {
                "m": 3,
                "w": 0.0,
                "prime": True,
                "indices": [0, 1, 2],
                "parent_link": self._sub_goal_0_links[0],
                "child_link": self._sub_goal_0_links[1],
                "desired_position": goal_position,
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
                "angle": goal_orientation,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        goal = GoalComposition(name="goal", contentDict=goal_dict)
        env.add_goal(goal)
        # Definition of the obstacle.
        radius_ring = 0.27
        obstacles = []
        rotation_matrix_ring = quaternionic.array(ring_orientation).to_rotation_matrix
        whole_position = goal.primeGoal().position()
        for i in range(self._obstacle_resolution + 1):
            angle = i/self._obstacle_resolution * 2.*np.pi
            origin_position = [
                -0.05,
                float(radius_ring * np.cos(angle)),
                float(radius_ring * np.sin(angle)),
            ]
            position = np.dot(np.transpose(rotation_matrix_ring), origin_position) + whole_position
            static_obst_dict = {
                "dim": 3,
                "type": "sphere",
                "geometry": {"position": position.tolist(), "radius": 0.08},
            }
            obstacles.append(SphereObstacle(name="staticObst", contentDict=static_obst_dict))
        for obst in obstacles:
            env.add_obstacle(obst)
        return env, obstacles, goal

    def evaluate_distance_to_goal(self, q: np.ndarray):
        sub_goal_0_position = np.array(self._goal.subGoals()[0].position())
        fk = self._generic_fk.fk(q, self._goal.subGoals()[0].parentLink(), self._goal.subGoals()[0].childLink(), positionOnly=True)
        return np.linalg.norm(sub_goal_0_position - fk) / self._initial_distance_to_goal_0 


    def set_goal_arguments(self, q0: np.ndarray, goal:GoalComposition, arguments):
        self._goal = goal
        sub_goal_0_position = np.array(goal.subGoals()[0].position())
        sub_goal_1_position = np.array(goal.subGoals()[1].position())
        sub_goal_1_quaternion = quaternionic.array(goal.subGoals()[1].angle())
        sub_goal_1_rotation_matrix = sub_goal_1_quaternion.to_rotation_matrix
        fk_0 = self._generic_fk.fk(q0, goal.subGoals()[0].parentLink(), goal.subGoals()[0].childLink(), positionOnly=True)
        self._initial_distance_to_goal_0 = np.linalg.norm(sub_goal_0_position - fk_0)
        #self._initial_distance_to_goal_0 = 1.0
        arguments['x_goal_0'] = sub_goal_0_position
        arguments['x_goal_1'] = sub_goal_1_position
        arguments['angle_goal_1'] = sub_goal_1_rotation_matrix
        arguments['weight_goal_0']=np.array([1.0])
        arguments['weight_goal_1']=np.array([5.0])
        return self._initial_distance_to_goal_0

