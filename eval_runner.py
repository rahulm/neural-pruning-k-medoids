"""
Only run evaluations on the given experiments for the given finetuned checkpoints.
"""

import csv
import os
from datetime import datetime
from typing import List, Text, Tuple

import eval_model


def evaluate_model(model_path: Text) -> Tuple[int, float, float]:
    model_size: int = os.path.getsize(model_path)
    model_train_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
        model_path_checkpoint=model_path,
        dataset_name="mnist",
        batch_size=2048,
        use_gpu=True,
        split="train",
    )
    model_test_acc, _ = eval_model.evaluate_model_from_checkpoint_file(
        model_path_checkpoint=model_path,
        dataset_name="mnist",
        batch_size=2048,
        use_gpu=True,
        split="test",
    )
    return model_size, model_train_acc, model_test_acc


def run_evals(experiment_name: Text, epochs_list: List[int]) -> None:
    """
    For each model checkpoint (epoch):
    - Get train and test accuracy on MNIST (eval_model).
    - Save info in a results csv, to be pasted into Google Sheets.
    """
    datetime_string: Text = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    experiments_folder_path: Text = os.path.join(
        "experiments", experiment_name, "pruning"
    )

    exp_names_list: List[Text] = [
        f.name for f in os.scandir(experiments_folder_path) if f.is_dir()
    ]
    exp_names_list.sort()

    results_list: List[List] = []

    try:
        for exp_name in exp_names_list:
            print("evaluating {}".format(exp_name))
            model_results: List = [exp_name]
            for epoch in epochs_list:
                print(">> epoch: {}".format(epoch))
                model_path: Text = os.path.join(
                    experiments_folder_path,
                    exp_name,
                    "finetuning",
                    "checkpoints",
                    "checkpoint-epoch_{}-model.pth".format(epoch),
                )
                size, train_acc, test_acc = evaluate_model(
                    model_path=model_path
                )
                model_results.extend(["", epoch, size, train_acc, test_acc])
                print(
                    "size: {}\t|\ttrain acc: {}\t|\ttest_acc: {}".format(
                        size, train_acc, test_acc
                    )
                )
            results_list.append(model_results)
            print("")
    finally:
        # Write results to csv
        print("writing results...")
        out_csv_path: Text = os.path.join(
            experiments_folder_path,
            "eval_results-{}.csv".format(datetime_string),
        )
        with open(out_csv_path, "w", newline="") as out_csv:
            csv_writer = csv.writer(out_csv)
            csv_writer.writerows(results_list)
        print("results at: {}".format(out_csv_path))


def main() -> None:
    # NOTE: Change these only.
    experiment_name: Text = "lenet_300_100-finetuned"
    epochs_list: List[int] = [
        0,
        1,
        2,
        3,
    ]
    run_evals(experiment_name=experiment_name, epochs_list=epochs_list)


if __name__ == "__main__":
    main()
