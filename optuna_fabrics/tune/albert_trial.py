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
from optuna_fabrics.planner.nonholonomic_symbolic_planner import NonHolonomicSymbolicFabricPlanner
from fabrics.planner.serialized_planner import SerializedFabricPlanner
import urdfenvs.albert_reacher

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
import quaternionic
 


class AlbertTrial(FabricsTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 10
        self._q0 = np.array([-2, 0, 0, 0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
        self._qdot0 = np.zeros(9)
        self._collision_links = ['base_link', 'panda_link8', 'panda_link4', "panda_link7", "panda_link5", "panda_hand"]
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(self._absolute_path + "/albert_fk.urdf", "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, 'origin', 'panda_hand')

    def initialize_environment(self, render=True, shuffle=True):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        env.add_obstacle(obstacles[1])
        steps the simulation once.
        """
        env = gym.make("albert-reacher-acc-v0", dt=0.05, render=render)
        return env


    def set_planner(self):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        serialize_file = self._absolute_path + "/planners/albert_planner.pkl"
        if os.path.exists(serialize_file):
            planner = SerializedFabricPlanner(serialize_file)
            return planner
        robot_type = "albert"

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
        planner = NonHolonomicSymbolicFabricPlanner(
            self._degrees_of_freedom,
            robot_type,
            urdf=self._urdf,
            root_link='origin',
            end_link=['panda_hand'],
        )
        albert_limits = [
                [-5, 5],
                [-5, 5],
                [-np.pi * 4, np.pi * 4],
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8974, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973]
            ]
        self_collision_pairs = {}
        # The planner hides all the logic behind the function set_components.
        goal = self.dummy_goal()
        planner.set_components(
            self._collision_links,
            self_collision_pairs,
            goal,
            number_obstacles=self._number_obstacles,
            limits=albert_limits,
        )
        planner.concretize()
        planner.serialize(serialize_file)
        return planner

    def manual_parameters(self) -> dict:
        return {
            "exp_geo_obst_leaf": 3,
            "k_geo_obst_leaf": 0.1,
            "exp_fin_obst_leaf": 3,
            "k_fin_obst_leaf": 0.1,
            "exp_geo_limit_leaf": 2,
            "k_geo_limit_leaf": 0.1,
            "exp_fin_limit_leaf": 3,
            "k_fin_limit_leaf": 0.05,
            "weight_attractor": 2,
            "base_inertia": 0.2,
            "alpha_b_damper" : 0.5,
            "beta_distant_damper" : 0.01,
            "beta_close_damper" : 6.5,
            "radius_shift_damper" : 0.2,
            "m_arm" : 1.0,
            "m_base" : 1.0,
            "m_rot": 0.1,
        }


    def sample_fabrics_params_uniform(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        exp_geo_obst_leaf = trial.suggest_int("exp_geo_obst_leaf", 1, 5, log=False)
        k_geo_obst_leaf = trial.suggest_float("k_geo_obst_leaf", 0.1, 1, log=True)
        exp_fin_obst_leaf = trial.suggest_int("exp_fin_obst_leaf", 1, 5, log=False)
        k_fin_obst_leaf = trial.suggest_float("k_fin_obst_leaf", 0.1, 1, log=True)
        exp_geo_limit_leaf = trial.suggest_int("exp_geo_limit_leaf", 1, 1, log=False)
        k_geo_limit_leaf = trial.suggest_float("k_geo_limit_leaf", 0.01, 0.2, log=True)
        exp_fin_limit_leaf = trial.suggest_int("exp_fin_limit_leaf", 1, 5, log=False)
        k_fin_limit_leaf = trial.suggest_float("k_fin_limit_leaf", 0.01, 0.2, log=True)
        #weight_attractor = trial.suggest_float("weight_attractor", 1.0, 2.0, log=False)
        base_inertia = trial.suggest_float("base_inertia", 0.01, 1.0, log=False)
        alpha_b_damper = trial.suggest_float('alpha_b_damper', 0, 1.0, log=False)
        beta_distant_damper = trial.suggest_float('beta_distant_damper', 0, 1.0, log=False)
        beta_close_damper = trial.suggest_float('beta_close_damper', 5, 20.0, log=False)
        radius_shift_damper = trial.suggest_float('radius_shift_damper', 0.01, 0.1, log=False)
        m_arm = trial.suggest_float("m_arm", 0.10, 5, log=False)
        m_base = trial.suggest_float("m_base", 0.10, 5, log=False)
        m_rot = trial.suggest_float("m_rot", 0.01, 1.0, log=False)
        return {
            "exp_geo_obst_leaf": exp_geo_obst_leaf,
            "k_geo_obst_leaf": k_geo_obst_leaf,
            "exp_fin_obst_leaf": exp_fin_obst_leaf,
            "k_fin_obst_leaf": k_fin_obst_leaf,
            "exp_geo_limit_leaf": exp_geo_limit_leaf,
            "k_geo_limit_leaf": k_geo_limit_leaf,
            "exp_fin_limit_leaf": exp_fin_limit_leaf,
            "k_fin_limit_leaf": k_fin_limit_leaf,
            #"weight_attractor": weight_attractor,
            "base_inertia": base_inertia,
            "alpha_b_damper": alpha_b_damper,
            "beta_close_damper": beta_close_damper,
            "radius_shift_damper": radius_shift_damper,
            "beta_distant_damper": beta_distant_damper,
            "m_arm": m_arm,
            "m_base": m_base,
            "m_rot": m_rot,
        }

    def run(self, params, planner: NonHolonomicSymbolicFabricPlanner, obstacles, ob, goal: GoalComposition, env, n_steps=1000):
        # Start the simulation
        logging.info("Starting simulation")
        q0 = ob['joint_state']['position']
        arguments, initial_distance_to_goal = self.set_goal_arguments(q0, goal)
        # sub_goal_0_position = np.array(goal.subGoals()[0].position())
        objective_value = 0.0
        distances_to_goal = []
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
        arguments['m_arm'] = np.array([params['m_arm']])
        arguments['m_base_x'] = np.array([params['m_base']])
        arguments['m_base_y'] = np.array([params['m_base']])
        arguments['m_rot'] = np.array([params['m_rot']])
        for _ in range(n_steps):
            qudot = np.array([ob['joint_state']['forward_velocity'][0], ob['joint_state']['velocity'][2]])
            qudot = np.concatenate((qudot, ob['joint_state']['velocity'][3:]))
            action = planner.compute_action(
                q=ob["joint_state"]['position'],
                qdot=ob["joint_state"]['velocity'],
                qudot=qudot,
                weight_goal_0=np.array([1.0]),
                weight_goal_1=np.array([5.0]),
                radius_body_base_link=np.array([0.5]),
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
            distances_to_goal.append(self.evaluate_distance_to_goal(q))
            distance_to_obstacles = []
            fk = self._generic_fk.fk(q, 'panda_link0', 'panda_hand', positionOnly=True)
            for obst in obstacles:
                distance_to_obstacles.append(np.linalg.norm(np.array(obst.position()) - fk))
            distance_to_obstacle += np.min(distance_to_obstacles)
        costs = {
            "path_length": path_length/initial_distance_to_goal,
            "time_to_goal": np.mean(np.array(distances_to_goal)),
            "obstacles": 1/distance_to_obstacle/n_steps
        }
        return self.total_costs(costs)


    def q0(self):
        return self._q0



