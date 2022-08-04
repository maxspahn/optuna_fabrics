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
from urdfenvs.generic_urdf_reacher.envs.acc import GenericUrdfReacherAccEnv

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
import quaternionic
 


class PandaTrial(FabricsTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 7
        self._q0 = np.array([0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
        self._qdot0 = np.zeros(7)
        self._collision_links = ['panda_link8', 'panda_link4', "panda_link7", "panda_link5", "panda_hand"]
        self._self_collision_pairs = {
            "panda_hand": ['panda_link2', 'panda_link4'], 
        }
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        self._urdf_file = self._absolute_path + "/panda.urdf"
        with open(self._urdf_file, "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, 'panda_link0', ['panda_hand', 'panda_link5_offset'])

    def initialize_environment(self, render=True):
        """
        Initializes the simulation environment.
        """
        env: GenericUrdfReacherAccEnv = gym.make(
            "generic-urdf-reacher-acc-v0", dt=0.05, urdf=self._urdf_file, render=render
        )
        return env


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
            root_link='panda_link0',
            end_link=['panda_hand'],
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

    @abstractmethod
    def set_goal_arguments(self, q0: np.ndarray, goal: GoalComposition):
        pass


    def run(self, params, planner: SymbolicFabricPlanner, obstacles, ob, goal: GoalComposition, env, n_steps=1000):
        # Start the simulation
        logging.info("Starting simulation")
        q0 = ob['joint_state']['position']
        arguments, initial_distance_to_goal = self.set_goal_arguments(q0, goal)
        # sub_goal_0_position = np.array(goal.subGoals()[0].position())
        objective_value = 0.0
        distance_to_goal = 0.0
        distance_to_obstacle = 0.0
        path_length = 0.0
        x_old = q0
        self.set_parameters(arguments, obstacles, params)
        for _ in range(n_steps):
            action = planner.compute_action(
                q=ob["joint_state"]['position'],
                qdot=ob["joint_state"]['velocity'],
                weight_goal_0=np.array([1.0]),
                weight_goal_1=np.array([5.0]),
                radius_body_panda_link4=np.array([0.1]),
                radius_body_panda_link5=np.array([0.08]),
                radius_body_panda_link7=np.array([0.08]),
                radius_body_panda_link8=np.array([0.1]),
                radius_body_panda_hand=np.array([0.1]),
                **arguments,
            )
            if np.linalg.norm(action) < 1e-5 or np.linalg.norm(action) > 1e3:
                action = np.zeros(7)
            warnings.filterwarnings("error")
            try:
                ob, *_ = env.step(action)
            except Exception as e:
                logging.warning(e)
                return 100
            q = ob['joint_state']['position']
            path_length += np.linalg.norm(q - x_old)
            x_old = q
            distance_to_goal += self.evaluate_distance_to_goal(q)
            distance_to_obstacles = []
            fk = self._generic_fk.fk(q, 'panda_link0', 'panda_hand', positionOnly=True)
            for obst in obstacles:
                distance_to_obstacles.append(np.linalg.norm(np.array(obst.position()) - fk))
            distance_to_obstacle += np.min(distance_to_obstacles)
        costs = {
            "path_length": path_length/initial_distance_to_goal,
            "time_to_goal": distance_to_goal/n_steps,
            "obstacles": 1/distance_to_obstacle/n_steps
        }
        return self.total_costs(costs)


    def q0(self):
        return self._q0



