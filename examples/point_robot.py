import sys
import os
import numpy as np
import gym
import logging
import warnings
import joblib
import pybullet

from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
logging.basicConfig(level=logging.ERROR)

class Autotuner(object):
    def __init__(self, render=False):
        self._render = render
        self._n_steps = 10000
        self.init_env()
        self.set_planner()
        self.parameters = {
            "x_goal_0":self._goal.sub_goals()[0].position(),
            "x_obst_0":self._obstacles[0].position()[0:2],
            "radius_obst_0":self._obstacles[0].radius(),
            "x_obst_1":self._obstacles[1].position()[0:2],
            "radius_obst_1":self._obstacles[1].radius(),
            "radius_body_1":0.2,
        }

    def init_env(self):
        robots = [
            GenericUrdfReacher(urdf="point_robot.urdf", mode="acc"),
        ]
        self.env: UrdfEnv  = gym.make(
            "urdf-env-v0",
            dt=0.01, robots=robots, render=self._render
        )
        # Definition of the obstacle.
        static_obst_dict_1 = {
                "type": "sphere",
                "geometry": {"position": [2.0, -0.3, 0.0], "radius": 0.8},
        }
        static_obst_dict_2 = {
                "type": "sphere",
                "geometry": {"position": [2.5, 1.8, 0.0], "radius": 0.6},
        }
        self._obstacles = [
            SphereObstacle(name="staticObst1", content_dict=static_obst_dict_1),
            SphereObstacle(name="staticObst1", content_dict=static_obst_dict_2),
        ]
        # Definition of the goal.
        goal_dict = {
                "subgoal0": {
                    "weight": 0.5,
                    "is_primary_goal": True,
                    "indices": [0, 1],
                    "parent_link" : 0,
                    "child_link" : 1,
                    "desired_position": [3.5, 0.5],
                    "epsilon" : 0.1,
                    "type": "staticSubGoal"
                }
        }
        self._goal = GoalComposition(name="goal", content_dict=goal_dict)

    def reset(self):
        self.env.reset()
        self.env.add_goal(self._goal.sub_goals()[0])
        self.env.add_obstacle(self._obstacles[0])
        self.env.add_obstacle(self._obstacles[1])

    def set_planner(self):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        Params
        ----------
        goal: StaticSubGoal
            The goal to the motion planning problem.
        """
        degrees_of_freedom = 2
        robot_type = "pointRobot"
        # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
        collision_geometry = "-sym('k_geo') / (x ** sym('exp_geo')) * xdot ** 2"
        collision_finsler = "sym('k_fin')/(x**sym('exp_fin')) * (1 - ca.heaviside(xdot))* xdot**2"
        self._planner = ParameterizedFabricPlanner(
                degrees_of_freedom,
                robot_type,
                collision_geometry=collision_geometry,
                collision_finsler=collision_finsler
        )
        collision_links = [1]
        self_collision_links = {}
        # The planner hides all the logic behind the function set_components.
        self._planner.set_components(
            collision_links,
            self_collision_links,
            self._goal,
            number_obstacles=2,
        )
        self._planner.concretize()

    def suggest_parameters(self, trial):
        self.parameters['k_geo_obst_0_1_leaf'] = trial.suggest_float('k_geo_obst_0_1_leaf', 0, 5)
        self.parameters['exp_geo_obst_0_1_leaf'] = trial.suggest_float('exp_geo_obst_0_1_leaf', 0, 5)
        self.parameters['k_fin_obst_0_1_leaf'] = trial.suggest_float('k_fin_obst_0_1_leaf', 0, 5)
        self.parameters['exp_fin_obst_0_1_leaf'] = trial.suggest_float('exp_fin_obst_0_1_leaf', 0, 5)
        self.parameters['k_geo_obst_1_1_leaf'] = trial.suggest_float('k_geo_obst_1_1_leaf', 0, 5)
        self.parameters['exp_geo_obst_1_1_leaf'] = trial.suggest_float('exp_geo_obst_1_1_leaf', 0, 5)
        self.parameters['k_fin_obst_1_1_leaf'] = trial.suggest_float('k_fin_obst_1_1_leaf', 0, 5)
        self.parameters['exp_fin_obst_1_1_leaf'] = trial.suggest_float('exp_fin_obst_1_1_leaf', 0, 5)
        self.parameters["weight_goal_0"] = trial.suggest_float('weight_goal_0', 0, 3)


    def objective(self, trial, tune=True):
        self.reset()
        if tune:
            self.suggest_parameters(trial)
        ob, *_ = self.env.step(np.zeros(3))
        action = np.zeros(3)
        distance = 0.0
        path_length = 0.0
        pos_old = np.zeros(2)
        initial_distance = np.linalg.norm(
                self._goal.sub_goals()[0].position() - 
                np.zeros(2)
        )
        for _ in range(self._n_steps):
            ob_robot = ob['robot_0']
            action[0:2] = self._planner.compute_action(
                q=ob_robot["joint_state"]["position"][0:2],
                qdot=ob_robot["joint_state"]["velocity"][0:2],
                **self.parameters
            )
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                ob, *_, = self.env.step(action)
            distance += np.linalg.norm(
                self._goal.sub_goals()[0].position() - 
                ob_robot['joint_state']['position'][0:2]
            )
            path_length += np.linalg.norm(
                ob_robot['joint_state']['position'][0:2] -
                pos_old
            )
            pos_old = ob_robot['joint_state']['position'][0:2]

        distance /= self._n_steps
        path_length /= initial_distance
        return distance + path_length




def optimize():
    autotuner = Autotuner()
    n_trials = 30
    print(f"Running {n_trials} trials...")
    study = optuna.create_study()
    study.optimize(autotuner.objective, n_trials=n_trials)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    joblib.dump(study, 'point_robot_study.pkl')

    plot_optimization_history(study).show()
    plot_param_importances(study).show()

def evaluate(trial_id):
    autotuner = Autotuner(render=True)
    study = joblib.load('point_robot_study.pkl')
    plot_optimization_history(study).show()
    plot_param_importances(study).show()
    autotuner.parameters.update(study.trials[trial_id].params)
    #pybullet.resetDebugVisualizerCamera(2, 0, 260, [2, 0.5, 0])
    pybullet.resetDebugVisualizerCamera(2, 0, 270.1, [2, 0.5, 0])
    

    autotuner.objective(study.trials[trial_id], tune=False)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        trial_id = int(sys.argv[1])
        evaluate(trial_id)
    else:
        optimize()
