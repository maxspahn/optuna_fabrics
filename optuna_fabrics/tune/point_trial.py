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

import planarenvs.point_robot


logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)



class PointTrial(FabricsTrial):
    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 2
        self._q0 = np.array([-8.0, 0.1])
        self._qdot0 = np.array([0.0, 0.0])
        self._number_obstacles = 5
        self._weights = {"path_length": 0.1, "time_to_goal": 0.7, "obstacles": 0.2}

    def initialize_environment(self, render=True, shuffle=True):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        env.add_obstacle(obstacles[1])
        steps the simulation once.
        """
        env = gym.make("point-robot-acc-v0", dt=0.01, render=render)
        initial_observation = env.reset(pos=self._q0, vel=self._qdot0)
        # Definition of the obstacle.
        static_obst_dict = {
            "dim": 2,
            "type": "sphere",
            "geometry": {"position": [0.0, 0.0], "radius": 1.0},
            "low": {'position': [-6.0, -6.0], 'radius': 0.3},
            "high": {'position': [6.0, 6.0], 'radius': 1.5},
        }
        obstacles = []
        for i in range(self._number_obstacles):
            obst_i = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
            if shuffle:
                obst_i.shuffle()
            obstacles.append(obst_i)
        # Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "m": 2,
                "w": 1.0,
                "prime": True,
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
        goal = GoalComposition(name="goal", contentDict=goal_dict)
        if shuffle:
            goal.shuffle()
        env.add_goal(goal)
        for obst in obstacles:
            env.add_obstacle(obst)
        return env, obstacles, goal, initial_observation

    def dummy_goal(self):
        goal_dict = {
            "subgoal0": {
                "m": 2,
                "w": 1.0,
                "prime": True,
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
        goal = GoalComposition(name="goal", contentDict=goal_dict)
        return goal


    def manual_parameters(self) -> dict:
        return {
            "exp_geo_obst_leaf": 2,
            "k_geo_obst_leaf": 3,
            "exp_fin_obst_leaf": 2,
            "k_fin_obst_leaf": 3,
            "weight_attractor": 1,
            "base_inertia": 0.5,
        }


    def set_planner(self):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        robot_type = "pointRobot"

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
        planner = SymbolicFabricPlanner(self._degrees_of_freedom, robot_type)
        # The planner hides all the logic behind the function set_components.
        collision_links = [1]
        self_collision_links = {}
        limits = [[-5, 5], [-5, 5]]
        goal = self.dummy_goal()
        planner.set_components(
            collision_links, self_collision_links, goal, number_obstacles=self._number_obstacles, limits=limits,
        )
        planner.concretize()
        return planner


    def objective(self, params, planner: SymbolicFabricPlanner, obstacles, ob, goal, env, n_steps=5000, shuffle=False):

        # Start the simulation
        print("Starting simulation")
        sub_goal_0_position = np.array(goal.subGoals()[0].position())
        initial_distance_to_goal = np.linalg.norm(sub_goal_0_position - ob['x'])
        # sub_goal_0_position = np.array(goal.subGoals()[0].position())
        sub_goal_0_weight = np.array(goal.subGoals()[0].weight())
        objective_value = 0.0
        distance_to_goal = 0.0
        distance_to_obstacle = 0.0
        path_length = 0.0
        x_old = ob['x']
        arguments = {}
        for i in range(self._number_obstacles):
            arguments[f"x_obst_{i}"] = np.array(obstacles[i].position())
            arguments[f"radius_obst_{i}"] = np.array(obstacles[i].radius())
            arguments[f"exp_geo_obst_{i}_1_leaf"] = np.array([params['exp_geo_obst_leaf']])
            arguments[f"k_geo_obst_{i}_1_leaf"] = np.array([params['k_geo_obst_leaf']])
            arguments[f"exp_fin_obst_{i}_1_leaf"] = np.array([params['exp_fin_obst_leaf']])
            arguments[f"k_fin_obst_{i}_1_leaf"] = np.array([params['k_fin_obst_leaf']])
        for i in range(self._degrees_of_freedom):
            for j in range(2):
                arguments[f"exp_limit_fin_limit_joint_{i}_{j}_leaf"] = np.array([params['exp_fin_limit_leaf']])
                arguments[f"exp_limit_geo_limit_joint_{i}_{j}_leaf"] = np.array([params['exp_geo_limit_leaf']])
                arguments[f"k_limit_fin_limit_joint_{i}_{j}_leaf"] = np.array([params['k_fin_limit_leaf']])
                arguments[f"k_limit_geo_limit_joint_{i}_{j}_leaf"] = np.array([params['k_geo_limit_leaf']])
        for _ in range(n_steps):
            action = planner.compute_action(
                q=ob["x"],
                qdot=ob["xdot"],
                x_goal_0=sub_goal_0_position,
                weight_goal_0=np.array([params['weight_attractor']]),
                radius_body_1=np.array([0.1]),
                base_inertia=np.array([params['base_inertia']]),
                **arguments,
            )
            if np.linalg.norm(action) < 1e-5 or np.linalg.norm(action) > 1e3:
                action = np.zeros(2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ob, *_ = env.step(action)
            path_length += np.linalg.norm(ob['x'] - x_old)
            x_old = ob['x']
            distance_to_goal += np.linalg.norm(sub_goal_0_position - ob["x"]) / initial_distance_to_goal
            distance_to_obstacles = []
            for obst in obstacles:
                distance_to_obstacles.append(np.linalg.norm(np.array(obst.position()) - ob['x']))
            distance_to_obstacle += np.min(distance_to_obstacles)
        costs = {
            "path_length": path_length/initial_distance_to_goal,
            "time_to_goal": distance_to_goal/n_steps,
            "obstacles": 1/distance_to_obstacle/n_steps
        }
        return sum([self._weights[i] * costs[i] for i in self._weights])

    def q0(self) -> np.ndarray:
        return self._q0

