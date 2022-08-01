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
        self._q0 = np.array([3.14, -1.0, -1.57, 0.1, 1.57, 0])
        self._qdot0 = np.zeros(self._degrees_of_freedom)
        self._collision_links = ['shoulder_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'forearm_link']
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        self._urdf_file = self._absolute_path + "/ur5.urdf"
        with open(self._urdf_file, "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, 'base_link', 'wrist_3_link')

    def initialize_environment(self, render=True, shuffle=True):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        env.add_obstacle(obstacles[1])
        steps the simulation once.
        """
        env = gym.make(
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
        serialize_file = self._absolute_path + "/planners/ur5_planner.pkl"
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
            root_link='base_link',
            end_link=['ee_link'],
        )
        ur5_limits= [
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-1 * np.pi, 1 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
                [-2 * np.pi, 2 * np.pi],
            ]
        self_collision_pairs = {}
        # The planner hides all the logic behind the function set_components.
        goal = self.dummy_goal()
        planner.set_components(
            self._collision_links,
            self_collision_pairs,
            goal,
            number_obstacles=self._number_obstacles,
            limits=ur5_limits,
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
        for j in self._collision_links:
            for i in range(self._number_obstacles):
                arguments[f"x_obst_{i}"] = np.array(obstacles[i].position())
                arguments[f"radius_obst_{i}"] = np.array(obstacles[i].radius())
                arguments[f"exp_geo_obst_{i}_{j}_leaf"] = np.array([params['exp_geo_obst_leaf']])
                arguments[f"k_geo_obst_{i}_{j}_leaf"] = np.array([params['k_geo_obst_leaf']])
                arguments[f"exp_fin_obst_{i}_{j}_leaf"] = np.array([params['exp_fin_obst_leaf']])
                arguments[f"k_fin_obst_{i}_{j}_leaf"] = np.array([params['k_fin_obst_leaf']])
        for j in range(self._degrees_of_freedom):
            for i in range(2):
                arguments[f"exp_limit_fin_limit_joint_{j}_{i}_leaf"] = np.array([params['exp_fin_limit_leaf']])
                arguments[f"exp_limit_geo_limit_joint_{j}_{i}_leaf"] = np.array([params['exp_geo_limit_leaf']])
                arguments[f"k_limit_fin_limit_joint_{j}_{i}_leaf"] = np.array([params['k_fin_limit_leaf']])
                arguments[f"k_limit_geo_limit_joint_{j}_{i}_leaf"] = np.array([params['k_geo_limit_leaf']])
        # damper arguments
        arguments['alpha_b_damper'] = np.array([params['alpha_b_damper']])
        arguments['beta_close_damper'] = np.array([params['beta_close_damper']])
        arguments['beta_distant_damper'] = np.array([params['beta_distant_damper']])
        arguments['radius_shift_damper'] = np.array([params['radius_shift_damper']])
        arguments['base_inertia'] = np.array([params['base_inertia']])
        for _ in range(n_steps):
            action = planner.compute_action(
                q=ob["joint_state"]['position'],
                qdot=ob["joint_state"]['velocity'],
                weight_goal_0=np.array([1.00]),
                weight_goal_1=np.array([5.00]),
                radius_body_wrist_1_link=np.array([0.05]),
                radius_body_wrist_2_link=np.array([0.05]),
                radius_body_wrist_3_link=np.array([0.05]),
                radius_body_shoulder_link=np.array([0.10]),
                radius_body_forearm_link=np.array([0.10]),
                **arguments,
            )
            #action = np.ones(6) * 0.1
            #action = np.clip(action, -1 * np.ones(6), np.ones(6))
            if np.linalg.norm(action) < 1e-5 or np.linalg.norm(action) > 1e3:
                action = np.zeros(6)
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
            fk = self._generic_fk.fk(q, 'base_link', 'ee_link', positionOnly=True)
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



