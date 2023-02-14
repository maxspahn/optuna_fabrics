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
from optuna_fabrics.tune.panda_trial import PandaTrial
 

logging.basicConfig(level=logging.INFO)
#optuna.logging.set_verbosity(optuna.logging.INFO)


class PandaReachTrial(PandaTrial):

    def __init__(self, weights = None):
        super().__init__(weights = weights)
        self._q0 = np.array([0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
        self._qdot0 = np.zeros(7)
        self._number_obstacles = 5
        self._collision_links = ['panda_link8', 'panda_link4', "panda_hand"]
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(self._absolute_path + "/panda_vacuum.urdf", "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, 'panda_link0', ['panda_vacuum', 'panda_vacuum_2'])

    def initialize_environment(self, render=True, shuffle=True):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        env.add_obstacle(obstacles[1])
        steps the simulation once.
        """
        env = gym.make("panda-reacher-acc-v0", dt=0.05, render=render)
        initial_observation = env.reset(pos=self._q0)
        static_obst_dict = {
            "dim": 3,
            "type": "sphere",
            "geometry": {"position": [1.0, 0.5, 0.4], "radius": 0.1},
            "low": {'position': [0.2, -0.8, 0.0], 'radius': 0.05},
            "high": {'position': [1.0, 0.8, 1.0], 'radius': 0.2},
        }
        obstacles = []
        for i in range(self._number_obstacles):
            obst_i = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
            if shuffle:
                obst_i.shuffle()
            obstacles.append(obst_i)
        # Definition of the obstacle.
        goal_dict = {
            "subgoal0": {
                "m": 3,
                "w": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_hand",
                "desired_position": [0.5, 0.0, 0.5],
                "epsilon": 0.05,
                "low": [0.3, -0.5, 0.1],
                "high": [0.8, 0.5, 0.9],
                "type": "staticSubGoal",
            },
        }
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        if shuffle:
            goal.shuffle()
        env.add_goal(goal)
        for obst in obstacles:
            env.add_obstacle(obst)
        return (env, obstacles, goal, initial_observation)


    def set_planner(self, goal: GoalComposition):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        serialize_file = self._absolute_path + "/planners/panda_reach_planner.pkl"
        if os.path.exists(serialize_file):
            planner = SerializedFabricPlanner(serialize_file)
            return planner
        degrees_of_freedom = 7
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
            degrees_of_freedom,
            robot_type,
            urdf=self._urdf,
            root_link='panda_link0',
            end_link=['panda_vacuum'],
        )
        panda_limits = [
                [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            ]
        self_collision_pairs = {}
        # The planner hides all the logic behind the function set_components.
        planner.set_components(
            self._collision_links,
            self_collision_pairs,
            goal,
            number_obstacles=self._number_obstacles,
            limits=panda_limits,
        )
        planner.concretize()
        planner.serialize(serialize_file)
        return planner


    def set_goal_arguments(self, q0: np.ndarray, goal:GoalComposition):
        arguments = {}
        sub_goal_0_position = np.array(goal.subGoals()[0].position())
        arguments['x_goal_0'] = sub_goal_0_position
        fk = self._generic_fk.fk(q0, goal.subGoals()[0].parentLink(), goal.subGoals()[0].childLink(), positionOnly=True)
        initial_distance_to_goal = np.linalg.norm(sub_goal_0_position - fk)
        return arguments, initial_distance_to_goal


