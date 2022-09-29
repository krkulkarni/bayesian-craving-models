import pandas as pd
from sys import path
from sys import platform
import argparse

parser = argparse.ArgumentParser(description="Run decision model")
parser.add_argument(
    "--model",
    choices=["biased", "heuristic", "rw", "rwdecay", "rwrl"],
    type=str,
    help="Craving model to run",
    required=True,
)
parser.add_argument("--rerun", default=False, action="store_true")
parser.add_argument(
    "--cores", type=int, help="Number of cores to use for multiprocessing", default=4
)
parser.add_argument(
    "--draws", type=int, help="Number of samples to draw from posterior", default=1000
)
args = parser.parse_args()

## Set project root
if platform == "linux" or platform == "linux2":
    root_dir = "/mnt/synapse/projects/SlotsTasks/online/prolific-food-craving/"
elif platform == "darwin":
    root_dir = "/Volumes/synapse/projects/SlotsTasks/online/prolific-food-craving/"
elif platform == "win32":
    root_dir = "Z:/projects/SlotsTasks/online/prolific-food-craving/"
model_functions_path = f"{root_dir}/derivatives/decision/"
path.append(model_functions_path)

from utils import load_data
from bayesian_model_functions import Biased, Heuristic, RescorlaWagner, RWDecay, RWRL

if __name__ == "__main__":

    path_to_summary = f"{root_dir}/rawdata/clean_df_summary.csv"
    path_to_longform = f"{root_dir}/rawdata/clean_df_longform.csv"
    df_summary, longform = load_data.load_clean_dbs(path_to_summary, path_to_longform)
    netcdf_path = f"{root_dir}/derivatives/decision/output/"

    if args.model == "biased":
        model = Biased.Biased(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    elif args.model == "heuristic":
        model = Heuristic.Heuristic(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    elif args.model == "rw":
        model = RescorlaWagner.RW(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    elif args.model == "rwdecay":
        model = RWDecay.RWDecay(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    elif args.model == "rwrl":
        model = RWRL.RWRL(
            model_name=args.model,
            save_path=netcdf_path,
            summary=df_summary,
            longform=longform,
        )

    model.fit(jupyter=False, cores=args.cores, rerun=args.rerun, draws=args.draws)
    model.calc_Q_table()
    model.calc_bics()

