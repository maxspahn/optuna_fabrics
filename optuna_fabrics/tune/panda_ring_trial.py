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
from optuna_fabrics.tune.panda_trial import PandaTrial
import quaternionic
 

logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)

def generate_random_orientation(w_amp, z_amp):
    return [1 - np.random.random(1) * w_amp, 0.0, 0.0, -z_amp + np.random.random(1) * 2 * z_amp]


class PandaRingTrial(PandaTrial):

    def __init__(self, weights=None):
        self._obstacle_resolution = 10
        self._number_obstacles = self._obstacle_resolution
        super().__init__(weights=weights)

    def shuffle_env(self, env):
        # Definition of the goal.
        goal_orientation = generate_random_orientation(0.4, 0.3)
        goal_dict = {
            "subgoal0": {
                "m": 3,
                "w": 0.0,
                "prime": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_hand",
                "desired_position": [0.75, 0.0, 0.55],
                "high": [0.7, 0.2, 0.7],
                "low": [0.4, -0.2, 0.5],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "m": 3,
                "w": 3.0,
                "prime": False,
                "indices": [0, 1, 2],
                "parent_link": "panda_link7",
                "child_link": "panda_hand",
                "desired_position": [0.107, 0.0, 0.0],
                "angle": goal_orientation,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        goal = GoalComposition(name="goal", contentDict=goal_dict)
        goal.shuffle()
        env.add_goal(goal)
        # Definition of the obstacle.
        radius_ring = 0.31
        obstacles = []
        ring_orientation = generate_random_orientation(0.5, 0.4)
        #goal_orientation = np.random.random(4).tolist()
        rotation_matrix_ring = quaternionic.array(ring_orientation).to_rotation_matrix
        whole_position = goal.primeGoal().position()
        for i in range(self._obstacle_resolution + 1):
            angle = i/self._obstacle_resolution * 2.*np.pi
            origin_position = [
                0.0,
                radius_ring * np.cos(angle),
                radius_ring * np.sin(angle),
            ]
            position = np.dot(np.transpose(rotation_matrix_ring), origin_position) + whole_position
            static_obst_dict = {
                "dim": 3,
                "type": "sphere",
                "geometry": {"position": position, "radius": 0.1},
            }
            obstacles.append(SphereObstacle(name="staticObst", contentDict=static_obst_dict))
        for obst in obstacles:
            env.add_obstacle(obst)
        return env, obstacles, goal


