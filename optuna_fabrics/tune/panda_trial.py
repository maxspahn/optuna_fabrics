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
 


class PandaTrial(FabricsTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 7
        self._q0 = np.array([0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
        self._qdot0 = np.zeros(7)
        self._collision_links = ['panda_link8', 'panda_link4', "panda_link7", "panda_link5", "panda_hand"]
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(self._absolute_path + "/panda_vacuum.urdf", "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, 'panda_link0', 'panda_hand')

    def initialize_environment(self, render=True, shuffle=True):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        env.add_obstacle(obstacles[1])
        steps the simulation once.
        """
        env = gym.make("panda-reacher-acc-v0", dt=0.05, render=render)
        env, obstacles, goal = self.shuffle_env(env)
        initial_observation = env.reset(pos=self._q0)
        return (env, obstacles, goal, initial_observation)


    def set_planner(self, goal: GoalComposition):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        serialize_file = self._absolute_path + "/planners/panda_planner.pkl"
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
        self_collision_pairs = {}
        # The planner hides all the logic behind the function set_components.
        planner.set_components(
            self._collision_links,
            self_collision_pairs,
            goal,
            number_obstacles=self._obstacle_resolution,
            limits=panda_limits,
        )
        planner.concretize()
        planner.serialize(serialize_file)
        return planner


    def run(self, params, planner: SymbolicFabricPlanner, obstacles, ob, goal: GoalComposition, env, n_steps=1000):
        # Start the simulation
        logging.info("Starting simulation")
        sub_goal_0_position = np.array(goal.subGoals()[0].position())
        sub_goal_0_weight= np.array(goal.subGoals()[0].weight())
        sub_goal_1_position = np.array(goal.subGoals()[1].position())
        sub_goal_1_weight= np.array(goal.subGoals()[1].weight())
        sub_goal_0_quaternion = quaternionic.array(goal.subGoals()[1].angle())
        sub_goal_0_rotation_matrix = sub_goal_0_quaternion.to_rotation_matrix
        fk = self._generic_fk.fk(ob['x'], goal.subGoals()[0].parentLink(), goal.subGoals()[0].childLink(), positionOnly=True)
        initial_distance_to_goal = np.linalg.norm(sub_goal_0_position - fk)
        # sub_goal_0_position = np.array(goal.subGoals()[0].position())
        objective_value = 0.0
        distance_to_goal = 0.0
        distance_to_obstacle = 0.0
        path_length = 0.0
        x_old = ob['x']
        arguments = {}
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
        for _ in range(n_steps):
            action = planner.compute_action(
                q=ob["x"],
                qdot=ob["xdot"],
                x_goal_0=sub_goal_0_position,
                x_goal_1=np.array([0.107, 0.0, 0.0]),
                weight_goal_0=np.array([params['weight_attractor']]),
                weight_goal_1=2 * np.array([params['weight_attractor']]),
                radius_body_panda_link4=np.array([0.1]),
                radius_body_panda_link5=np.array([0.08]),
                radius_body_panda_link7=np.array([0.08]),
                radius_body_panda_link8=np.array([0.1]),
                radius_body_panda_hand=np.array([0.1]),
                base_inertia=np.array([params['base_inertia']]),
                angle_goal_1=sub_goal_0_rotation_matrix,
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
            path_length += np.linalg.norm(ob['x'] - x_old)
            x_old = ob['x']
            
            fk = self._generic_fk.fk(ob['x'], goal.subGoals()[0].parentLink(), goal.subGoals()[0].childLink(), positionOnly=True)
            distance_to_goal += np.linalg.norm(sub_goal_0_position - fk) / initial_distance_to_goal
            distance_to_obstacles = []
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



