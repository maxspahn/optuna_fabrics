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

class PandaTableTrial(PandaTrial):

    def __init__(self, weights=None):
        self._number_obstacles = 5
        super().__init__(weights)

    def shuffle_env(self, env, shuffle=True):
        static_obst_dict = {
            "dim": 3,
            "type": "sphere",
            "geometry": {"position": [1000.0, 0.5, 0.1], "radius": 0.1},
            "low": {'position': [0.2, -0.7, 0.0], 'radius': 0.05},
            "high": {'position': [1.0, 0.7, 0.4], 'radius': 0.1},
        }
        obstacles = []
        for i in range(self._number_obstacles):
            obst_i = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
            if shuffle:
                obst_i.shuffle()
            obstacles.append(obst_i)
        goal_dict = {
            "subgoal0": {
                "m": 3,
                "w": 1.0,
                "prime": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_hand",
                "desired_position": [0.5, 0.0, 0.1],
                "epsilon": 0.05,
                "low": [0.4, -0.5, 0.05],
                "high": [0.8, 0.5, 0.05],
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "m": 3,
                "w": 3.0,
                "prime": False,
                "indices": [0, 1, 2],
                "parent_link": "panda_link7",
                "child_link": "panda_hand",
                "desired_position": [0.0, 0.0, 0.107],
                #"angle": [0, -np.sqrt(2), 0.0, np.sqrt(2)], # facing downwords
                "angle": [0, 1, 0, 0],
                "low": [0.0, 0.0, 0.107],
                "high": [0.0, 0.0, 0.107],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        goal = GoalComposition(name="goal", contentDict=goal_dict)
        if shuffle:
            goal.shuffle()
        env.add_goal(goal)
        for obst in obstacles:
            env.add_obstacle(obst)
        return env, obstacles, goal

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
        serialize_file = self._absolute_path + "/planners/panda_table_planner.pkl"
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


    def run(self, params, planner: SymbolicFabricPlanner, obstacles, ob, goal: GoalComposition, env, n_steps=1000):
        # Start the simulation
        print("Starting simulation")
        sub_goal_0_position = np.array(goal.subGoals()[0].position())
        fk_0 = self._generic_fk.fk(ob['x'], goal.subGoals()[0].parentLink(), goal.subGoals()[0].childLink(), positionOnly=True)
        initial_distance_to_goal_0 = np.linalg.norm(sub_goal_0_position - fk_0)
        fk_1_0 = self._generic_fk.fk(ob['x'], "panda_link0", goal.subGoals()[1].parentLink(), positionOnly=True)
        fk_1_1 = self._generic_fk.fk(ob['x'], "panda_link0", goal.subGoals()[1].childLink(), positionOnly=True)
        initial_distance_to_goal_1 = np.linalg.norm(fk_1_0[0:2] - fk_1_1[0:2])
        sub_goal_1_position = np.array(goal.subGoals()[1].position())
        sub_goal_1_quaternion = quaternionic.array(goal.subGoals()[1].angle())
        sub_goal_1_rotation_matrix = sub_goal_1_quaternion.to_rotation_matrix
        # sub_goal_0_position = np.array(goal.subGoals()[0].position())
        objective_value = 0.0
        distance_to_goal = 0.0
        distance_to_orientation = 0.0
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
                x_goal_1=sub_goal_1_position,
                angle_goal_1=sub_goal_1_rotation_matrix,
                weight_goal_1=2*np.array([params['weight_attractor']]),
                weight_goal_0=np.array([params['weight_attractor']]),
                radius_body_panda_link4=np.array([0.1]),
                radius_body_panda_link8=np.array([0.1]),
                radius_body_panda_hand=np.array([0.1]),
                base_inertia=np.array([params['base_inertia']]),
                **arguments,
            )
            if np.linalg.norm(action) < 1e-5 or np.linalg.norm(action) > 1e3:
                action = np.zeros(7)
            warnings.filterwarnings("error")
            try:
                ob, *_ = env.step(action)
            except Exception as e:
                print(e)
                return 100
            path_length += np.linalg.norm(ob['x'] - x_old)
            x_old = ob['x']
            sub_goal_0_weight = np.array(goal.subGoals()[0].weight())
            
            fk_0 = self._generic_fk.fk(ob['x'], goal.subGoals()[0].parentLink(), goal.subGoals()[0].childLink(), positionOnly=True)
            fk_1 = self._generic_fk.fk(ob['x'], goal.subGoals()[1].parentLink(), goal.subGoals()[1].childLink(), positionOnly=True)
            distance_position = np.linalg.norm(sub_goal_0_position - fk_0) / initial_distance_to_goal_0
            fk_1_0 = self._generic_fk.fk(ob['x'], "panda_link0", goal.subGoals()[1].parentLink(), positionOnly=True)
            fk_1_1 = self._generic_fk.fk(ob['x'], "panda_link0", goal.subGoals()[1].childLink(), positionOnly=True)
            distance_orientation = np.linalg.norm(fk_1_0[0:2] - fk_1_1[0:2])/ initial_distance_to_goal_1
            distance_to_goal += distance_position
            distance_to_orientation += distance_orientation
            distance_to_obstacles = []
            for obst in obstacles:
                distance_to_obstacles.append(np.linalg.norm(np.array(obst.position()) - fk_0))
            distance_to_obstacle += np.min(distance_to_obstacles)
        costs = {
            "path_length": path_length/initial_distance_to_goal_0,
            "time_to_goal": distance_to_goal/n_steps,
            "time_to_orientation": distance_to_orientation/n_steps,
            "obstacles": 1/distance_to_obstacle/n_steps
        }
        return sum([self._weights[i] * costs[i] for i in self._weights])


    def objective(self, trial, planner, obstacles, goal, env, q0):
        ob = env.reset(pos=q0)
        params = self.sample_fabrics_params_uniform(trial)
        return self.run(params, planner, obstacles, ob, goal, env)

    def q0(self):
        return self._q0



