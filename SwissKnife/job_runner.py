"""
SIPEC
MARKUS MARKS
JOB RUNNER
"""

import subprocess
import sys
from argparse import ArgumentParser

sys.path.append("../")


def main():
    args = parser.parse_args()
    operation = args.operation
    gpu_name = args.gpu

    if operation == "ablation_mouse_segmentation":
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        random_keys = [1, 2, 3]

        for random_key in random_keys:
            for fraction in fractions:
                print("running job")
                subprocess.call(
                    [
                        "python",
                        "./segmentation.py",
                        "--operation",
                        "mouse_ablation",
                        "--gpu",
                        gpu_name,
                        "--cv_folds",
                        "5",
                        "--random_seed",
                        str(random_key),
                        "--fraction",
                        str(fraction),
                    ],
                )
                print("job done")

    if operation == "ablation_mouse_behavior":
        fractions = [0.2, 0.4, 0.6, 0.8]
        random_keys = [1, 2, 3]

        for random_key in random_keys:
            for fraction in fractions:
                print("running job")
                subprocess.call(
                    [
                        "python",
                        "../classification_comparison.py",
                        "--config_name",
                        "behavior_config_final",
                        "--gpu",
                        "0",
                        "--cv_folds",
                        "5",
                        "--random_seed",
                        str(random_key),
                        "--fraction",
                        str(fraction),
                    ],
                )
                print("job done")

    print("done")


parser = ArgumentParser()
parser.add_argument(
    "--operation",
    action="store",
    dest="operation",
    type=str,
    default="train_primate",
    help="standard training options for SIPEC data",
)
parser.add_argument(
    "--gpu",
    action="store",
    dest="gpu",
    type=str,
    default=None,
    help="filename of the video to be processed (has to be a segmented one)",
)

if __name__ == "__main__":
    main()
