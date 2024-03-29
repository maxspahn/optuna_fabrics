"""
Optuna example that optimizes a simple quadratic function.

In this example, we optimize a simple quadratic function. We also demonstrate how to continue an
optimization and to use timeouts.

"""
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y

def optimize():
    # Let us minimize the objective function above.
    print("Running 10 trials...")
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    # We can continue the optimization as follows.
    # print("Running 20 additional trials...")
    # study.optimize(objective, n_trials=20)
    # print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    # We can specify the timeout instead of a number of trials.
    # print("Running additional trials in 2 seconds...")
    # study.optimize(objective, timeout=2.0)
    # print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    plot_optimization_history(study).show()
    plot_param_importances(study).show()
    #fig1.show()
    #fig2.show()



if __name__ == "__main__":
    optimize()
