import optuna
import numpy as np
import joblib
import sys
import logging
import csv

logging.basicConfig(level=logging.INFO)

study: optuna.Study = joblib.load(sys.argv[1])
trials = study.get_trials()
logging.info(f"number of trials in study {len(trials)}")
values = [trial.value for trial in trials]
logging.info(f"{values}")

best_value = 1.0

with open("study_values.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["trial", "value", "best_value"])
    for i, value in enumerate(values):
        best_value = min(best_value, value)
        writer.writerow([i, value, best_value])

