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


from optuna_fabrics.tune.fabrics_trial import FabricsTrial
import quaternionic

logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)

class PointAvoidanceTrial(FabricsTrial):
    def __init__(self, weights=None):
        self._number_obstacles = 5
        self._obstacle_resolution = self._number_obstacles
        super().__init__(weights=weights)

    def dummy_goal(self):
        goal_dict = {
            "subgoal0": {
                "m": 2,
                "w": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link": 0,
                "child_link": 2,
                "desired_position": [3.0, 0.0],
                "low": [1.0, -5.0],
                "high": [5.0, 5.0],
                "epsilon": 0.15,
                "type": "staticSubGoal",
            }
        }
        return GoalComposition(name="goal", content_dict=goal_dict)


    def shuffle_env(self, env, shuffle=True):
        static_obst_dict = {
            "dim": 2,
            "type": "sphere",
            "geometry": {"position": [0.0, 0.0], "radius": 0.1},
            "low": {'position': [-6.0, -6.0], 'radius': 0.3},
            "high": {'position': [6.0, 6.0], 'radius': 1.5},
        }
        obstacle_positions =[
            [-1.0, 0.1],
            [-0.2, -0.8],
            [0.3, 1.5],
            [1.0, 0.5],
            [2.0, -1.2],
        ]
        obstacles = []
        for i in range(self._number_obstacles):
            obst_i = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
            if shuffle:
                obst_i.shuffle()
            else:
                obst_i._config.geometry.position = obstacle_positions[i]
            obstacles.append(obst_i)
        # Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "m": 2,
                "w": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link": 0,
                "child_link": 2,
                "desired_position": [3.0, -0.2],
                "low": [1.0, -5.0],
                "high": [5.0, 5.0],
                "epsilon": 0.15,
                "type": "staticSubGoal",
            }
        }
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        if shuffle:
            goal.shuffle()
        env.add_goal(goal)
        for obst in obstacles:
            env.add_obstacle(obst)
        return env, obstacles, goal

    def evaluate_distance_to_goal(self, q: np.ndarray):
        sub_goal_0_position = np.array(self._goal.subGoals()[0].position())
        fk = self._fk.fk(q, self._goal.subGoals()[0].childLink(), positionOnly=True)
        return np.linalg.norm(sub_goal_0_position - fk) / self._initial_distance_to_goal_0 


    def set_goal_arguments(self, q0: np.ndarray, goal:GoalComposition, arguments):
        self._goal = goal
        sub_goal_0_position = np.array(goal.subGoals()[0].position())
        fk_0 = self._fk.fk(q0, goal.subGoals()[0].childLink(), positionOnly=True)
        self._initial_distance_to_goal_0 = np.linalg.norm(sub_goal_0_position - fk_0)
        #self._initial_distance_to_goal_0 = 1.0
        arguments['x_goal_0'] = sub_goal_0_position
        arguments['weight_goal_0']=np.array([1.0])
        arguments['angle_goal_1']=np.identity(3)
        return self._initial_distance_to_goal_0

