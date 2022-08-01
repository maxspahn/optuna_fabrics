from abc import abstractmethod
from typing import Dict, Any
from MotionPlanningGoal.goalComposition import GoalComposition
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
import optuna


class FabricsTrial(object):
    def __init__(self, weights = None):
        self._weights = {"path_length": 0.4, "time_to_goal": 0.4, "obstacles": 0.2}
        if weights:
            self._weights = weights

    @abstractmethod
    def initialize_environment(self, render=True, shuffle=True):
        pass

    @abstractmethod
    def set_planner(self, render=True):
        pass

    def total_costs(self, costs: dict):
        return sum([self._weights[i] * costs[i] for i in self._weights])

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
            "ex_factor": 50.0,
        }

    @abstractmethod
    def run(self, params, planner: SymbolicFabricPlanner, obstacles, ob, goal: GoalComposition, env, n_steps=5000):
        pass

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
        ex_factor = trial.suggest_float("ex_factor", 1.0, 30.0, log=False)
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
            "ex_factor": ex_factor,
        }

    @abstractmethod
    def q0(self):
        pass


    @abstractmethod
    def shuffle_env(self, env):
        pass



    def objective(self, trial, planner, env, q0):
        ob = env.reset(pos=q0)
        env, obstacles, goal = self.shuffle_env(env)
        params = self.sample_fabrics_params_uniform(trial)
        return self.run(params, planner, obstacles, ob, goal, env)
