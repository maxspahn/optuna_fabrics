import logging
import datetime
import optuna
import joblib
import csv
import argparse
from optuna.visualization import plot_optimization_history, plot_param_importances
import numpy as np
import matplotlib.pyplot as plt
import yaml



from optuna_fabrics.tune.fabrics_trial import FabricsTrial

logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)

class FabricsStudy(object):
    def __init__(self, trial: FabricsTrial):
        self.initialize_argument_parser()
        cli_arguments = self._parser.parse_args()
        self._trial = trial
        self._q0 = None
        self._number_trials = cli_arguments.number_trials
        self._input_file=cli_arguments.input
        self._output_file=cli_arguments.output
        self._evaluate = cli_arguments.evaluate
        self._manual_tuning = cli_arguments.manual_tuning
        self._render = cli_arguments.render
        self._shuffle = cli_arguments.shuffle
        self._random_parameters = cli_arguments.random_parameters
        self._video_name = cli_arguments.video_name
        self._trial_index = cli_arguments.trial
        if cli_arguments.seed >= 0:
            np.random.seed(cli_arguments.seed)
        if not self._manual_tuning:
            self.initialize_study()
        if not self._output_file:
            self._output_file = "temp_optuna_output.pkl"

    def initialize_argument_parser(self):
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument("--input", "-i", type=str, help="Study to be loaded")
        self._parser.add_argument("--output", "-o", type=str, help="Path to save study")
        self._parser.add_argument("--number_trials", "-n", type=int, default=10)
        self._parser.add_argument("--render", "-r", action='store_true')
        self._parser.add_argument("--evaluate", "-e", action='store_true')
        self._parser.add_argument("--manual_tuning", "-m", action="store_true")
        self._parser.add_argument("--no-shuffle", "-ns", action="store_false", dest="shuffle")
        self._parser.add_argument("--seed", "-s", type=int, default=-1)
        self._parser.add_argument("--random-parameters", "-rp", action="store_true", dest="random_parameters")
        self._parser.add_argument("--video-name", "-vn", type=str, help="Video file name", default=None)
        self._parser.add_argument("--trial", "-t", type=int, help="Select trial from study", default=None)
        self._parser.set_defaults(render=False, manual_tuning=False, evaluate=False, shuffle=True, random_parameters=False)

    def print_parameters(self, params: dict, header: str):
        params_string = f"{header}\n"
        for name, value in params.items():
            name_string = f"{name}".ljust(25)
            params_string += f"{name_string}: {value}\n"
        logging.info(params_string)


    def tune(self):
        # Let us minimize the objective function above.
        env = self._trial.initialize_environment(render=self._render)
        q0 = self._trial.q0()
        planner = self._trial.set_planner()
        logging.info(f"Running {self._number_trials} trials...")
        self._study.optimize(
            lambda trial: self._trial.objective(trial, planner, env, q0, shuffle=self._shuffle),
            n_trials=self._number_trials,
        )
        self.print_parameters(self._study.best_params, "Best parameters")
        logging.info("Saving study")
        self.save_study()

    def save_study(self):
        logging.info(f"Saving study to {self._output_file}")
        joblib.dump(self._study, self._output_file)

    def test_result(self):
        if self._manual_tuning:
            logging.info("Using manual tuning")
            params = self._trial.manual_parameters()
            params = self._trial.caspar_parameters()
        elif self._random_parameters:
            logging.info("Using random parameters")
            params = self._trial.random_parameters()
        else:
            logging.info("Using autotuned parameters")
            if self._trial_index:
                if self._trial_index > len(self._study.trials):
                    raise IndexError(f"There are only {len(self._study.trials)} "\
                        +"trials in the study. Your choice "\
                        +f"{self._trial_index} is out of range")
                params = self._study.trials[self._trial_index].params
            else:
                params = self._study.best_params
        self.print_parameters(params, "Selected parameters")
        total_costs = []
        env = self._trial.initialize_environment(render=self._render)
        planner = self._trial.set_planner()
        for i in range(self._number_trials):
            q0 = self._trial.q0()
            ob = env.reset(pos=q0)
            np.random.seed(0)
            env, obstacles, goal = self._trial.shuffle_env(env, shuffle=self._shuffle)
            self._trial.create_collision_metric(obstacles)
            for obst in obstacles:
                env.add_obstacle(obst)
            costs = self._trial.run(params, planner, obstacles, ob, goal, env)
            costs["total_costs"] = self._trial.total_costs(costs)
            logging.info(f"Finished test run {i} with cost: {costs}")
            total_costs.append(costs)
        time_stamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
        with open(f"temp_results/result_{time_stamp}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(total_costs[0].keys())
            for cost in total_costs:
                writer.writerow(list(cost.values()))
        with open(f"temp_results/params_{time_stamp}.yaml", "w") as f:
            yaml.dump(params, f)



    def show_history(self):
        fig1 = plot_optimization_history(self._study)
        fig2 = plot_param_importances(self._study)
        #fig.update_layout(
        #    font=dict(
        #        family="Serif",
        #        size=30,
        #        color="black"
        #    )
        #)
        fig1.show()
        fig2.show()

    def initialize_study(self) -> None:
        if self._input_file:
            logging.info(f"Reading study from {self._input_file}")
            self._study = joblib.load(self._input_file)
        else:
            logging.info(f"Creating new study")
            self._study = optuna.create_study(direction="minimize")

    def run(self):
        if not self._evaluate:
            self.tune()
        else:
            self.test_result()


