from abc import abstractmethod
from typing import Dict, Any
from MotionPlanningGoal.goalComposition import GoalComposition
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
import gym
import logging
import time
import warnings
from urdfenvs.generic_urdf_reacher.envs.acc import GenericUrdfReacherAccEnv
import optuna
import numpy as np
import casadi as ca


class FabricsTrial(object):
    def __init__(self, weights = None):
        self._weights = {"path_length": 0.4, "time_to_goal": 0.4, "obstacles": 0.2}
        self._maximum_seconds = 30
        self._dt = 0.05
        if weights:
            self._weights = weights

    def initialize_environment(self, render=True):
        """
        Initializes the simulation environment.
        """
        env: GenericUrdfReacherAccEnv = gym.make(
            "generic-urdf-reacher-acc-v0", dt=self._dt, urdf=self._urdf_file, render=render
        )
        return env

    @abstractmethod
    def set_planner(self, render=True):
        pass

    def total_costs(self, costs: dict):
        return sum([self._weights[i] * costs[i] for i in self._weights])

    def manual_parameters(self) -> dict:
        return {
            "exp_geo_obst_leaf": 3,
            "k_geo_obst_leaf": 0.03,
            "exp_fin_obst_leaf": 3,
            "k_fin_obst_leaf": 0.03,
            "exp_geo_self_leaf": 3,
            "k_geo_self_leaf": 0.03,
            "exp_fin_self_leaf": 3,
            "k_fin_self_leaf": 0.03,
            "exp_geo_limit_leaf": 2,
            "k_geo_limit_leaf": 0.3,
            "exp_fin_limit_leaf": 3,
            "k_fin_limit_leaf": 0.05,
            "weight_attractor": 2,
            "base_inertia": 0.20,
            "alpha_b_damper" : 0.5,
            "beta_distant_damper" : 0.01,
            "beta_close_damper" : 6.5,
            "radius_shift_damper" : 0.05,
            "ex_factor": 15.0,
        }

    def sample_fabrics_params_uniform(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        exp_geo_obst_leaf = trial.suggest_int("exp_geo_obst_leaf", 1, 5, log=False)
        k_geo_obst_leaf = trial.suggest_float("k_geo_obst_leaf", 0.01, 1, log=True)
        exp_fin_obst_leaf = trial.suggest_int("exp_fin_obst_leaf", 1, 5, log=False)
        k_fin_obst_leaf = trial.suggest_float("k_fin_obst_leaf", 0.01, 1, log=True)
        exp_geo_self_leaf = trial.suggest_int("exp_geo_self_leaf", 1, 5, log=False)
        k_geo_self_leaf = trial.suggest_float("k_geo_self_leaf", 0.01, 1, log=True)
        exp_fin_self_leaf = trial.suggest_int("exp_fin_self_leaf", 1, 5, log=False)
        k_fin_self_leaf = trial.suggest_float("k_fin_self_leaf", 0.01, 1, log=True)
        exp_geo_limit_leaf = trial.suggest_int("exp_geo_limit_leaf", 1, 1, log=False)
        k_geo_limit_leaf = trial.suggest_float("k_geo_limit_leaf", 0.01, 0.2, log=True)
        exp_fin_limit_leaf = trial.suggest_int("exp_fin_limit_leaf", 1, 5, log=False)
        k_fin_limit_leaf = trial.suggest_float("k_fin_limit_leaf", 0.01, 0.2, log=True)
        base_inertia = trial.suggest_float("base_inertia", 0.01, 1.0, log=False)
        alpha_b_damper = trial.suggest_float('alpha_b_damper', 0, 1.0, log=False)
        beta_distant_damper = trial.suggest_float('beta_distant_damper', 0, 1.0, log=False)
        beta_close_damper = trial.suggest_float('beta_close_damper', 5, 20.0, log=False)
        radius_shift_damper = trial.suggest_float('radius_shift_damper', 0.01, 0.1, log=False)
        ex_factor = trial.suggest_float("ex_factor", 1.0, 30.0, log=False)
        return {
            "exp_geo_obst_leaf": exp_geo_obst_leaf,
            "k_geo_obst_leaf": k_geo_obst_leaf,
            "exp_fin_obst_leaf": exp_fin_obst_leaf,
            "k_fin_obst_leaf": k_fin_obst_leaf,
            "exp_geo_self_leaf": exp_geo_self_leaf,
            "k_geo_self_leaf": k_geo_self_leaf,
            "exp_fin_self_leaf": exp_fin_self_leaf,
            "k_fin_self_leaf": k_fin_self_leaf,
            "exp_geo_limit_leaf": exp_geo_limit_leaf,
            "k_geo_limit_leaf": k_geo_limit_leaf,
            "exp_fin_limit_leaf": exp_fin_limit_leaf,
            "k_fin_limit_leaf": k_fin_limit_leaf,
            "base_inertia": base_inertia,
            "alpha_b_damper": alpha_b_damper,
            "beta_close_damper": beta_close_damper,
            "radius_shift_damper": radius_shift_damper,
            "beta_distant_damper": beta_distant_damper,
            "ex_factor": ex_factor,
        }

    def set_collision_arguments(self, arguments: dict):
        for link in self._collision_links:
            arguments[f"radius_body_{link}"] = np.array([self._link_sizes[link]])

    def set_parameters(self, arguments, obstacles, params):
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
        for link, paired_links in self._self_collision_pairs.items():
            for paired_link in paired_links:
                arguments[f"exp_self_fin_self_collision_{link}_{paired_link}"] = np.array([params['exp_fin_self_leaf']])
                arguments[f"exp_self_geo_self_collision_{link}_{paired_link}"] = np.array([params['exp_geo_self_leaf']])
                arguments[f"k_self_fin_self_collision_{link}_{paired_link}"] = np.array([params['k_fin_self_leaf']])
                arguments[f"k_self_geo_self_collision_{link}_{paired_link}"] = np.array([params['k_geo_self_leaf']])
        # damper arguments
        arguments['alpha_b_damper'] = np.array([params['alpha_b_damper']])
        arguments['beta_close_damper'] = np.array([params['beta_close_damper']])
        arguments['beta_distant_damper'] = np.array([params['beta_distant_damper']])
        arguments['radius_shift_damper'] = np.array([params['radius_shift_damper']])
        arguments['base_inertia'] = np.array([params['base_inertia']])
        arguments['ex_factor_damper'] = np.array([params['ex_factor']])


    @abstractmethod
    def shuffle_env(self, env):
        pass

    @abstractmethod
    def set_goal_arguments(self, q0: np.ndarray, goal:GoalComposition, arguments):
        pass

    def run(self, params, planner: SymbolicFabricPlanner, obstacles, ob, goal: GoalComposition, env):
        # Start the simulation
        logging.info("Starting simulation")
        q0 = ob['joint_state']['position']
        arguments = {}
        self.set_collision_arguments(arguments)
        self.set_goal_arguments(q0, goal, arguments)
        self.set_parameters(arguments, obstacles, params)
        initial_distance_to_obstacles = self.evaluate_distance_to_closest_obstacle(obstacles, q0)
        distances_to_goal = []
        distances_to_closest_obstacle = []
        path_length = 0.0
        x_old = q0
        while env.t() < self._maximum_seconds:
            t0 = time.perf_counter()
            action = planner.compute_action(
                q=ob["joint_state"]['position'],
                qdot=ob["joint_state"]['velocity'],
                **arguments,
            )
            t1 = time.perf_counter()
            if np.linalg.norm(action) < 1e-5 or np.linalg.norm(action) > 1e3:
                action = np.zeros(self._degrees_of_freedom)
            warnings.filterwarnings("error")
            try:
                ob, *_ = env.step(action)
            except Exception as e:
                logging.warning(e)
                return {"path_length": 1.0, "time_to_goal": 1.0, "obstacles": 1.0}
            q = ob['joint_state']['position']
            t2 = time.perf_counter()
            path_length += np.linalg.norm(q - x_old)
            x_old = q
            distances_to_goal.append(self.evaluate_distance_to_goal(q))
            distances_to_closest_obstacle.append(self.evaluate_distance_to_closest_obstacle(obstacles, q))
            t3 = time.perf_counter()
            logging.debug(f"Compute action {t1-t0}\nStepping environment: {t2-t1}\nCompute metrics{t3-t2}")
        costs = {
            "path_length": path_length/10,
            "time_to_goal": np.mean(np.array(distances_to_goal)),
            "obstacles": 1 - np.min(distances_to_closest_obstacle) / initial_distance_to_obstacles
        }
        return costs

    def create_collision_metric(self, obstacles):
        q = ca.SX.sym("q", self._degrees_of_freedom)
        distance_to_obstacles = 10000
        for link in self._collision_links:
            fk = self._generic_fk.fk(q, self._root_link, link, positionOnly=True)
            for obst in obstacles:
                obst_position = np.array(obst.position())
                distance_to_obstacles = ca.fmin(distance_to_obstacles, ca.norm_2(obst_position - fk))
        self._collision_metric = ca.Function("collision_metric", [q], [distance_to_obstacles])


    def evaluate_distance_to_closest_obstacle(self, obstacles, q: np.ndarray):
        casadi_metric = self._collision_metric(q)
        return casadi_metric

    def q0(self):
        return self._q0



    def objective(self, trial, planner, env, q0, shuffle=True):
        ob = env.reset(pos=q0)
        env, obstacles, goal = self.shuffle_env(env, randomize=shuffle)
        self.create_collision_metric(obstacles)
        params = self.sample_fabrics_params_uniform(trial)
        return self.total_costs(self.run(params, planner, obstacles, ob, goal, env))
