import argparse
import datetime
from pathlib import Path
from typing import List

import motmetrics as mm
from evaluator import Evaluator


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise ValueError(f"{data_dir} not found!")
    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise ValueError(f"{result_dir} not found!")

    accumulators: List[mm.MOTAccumulator] = []
    sequences: List[str] = []
    for seq_dir in data_dir.iterdir():
        if not (
            result_path := (result_dir / seq_dir.name).with_suffix(".txt")
        ).exists():
            raise ValueError(f"{result_path} not found!")
        evaluator = Evaluator(seq_dir)
        accumulators.append(evaluator.eval_file(result_path))
        sequences.append(seq_dir.name)

    metrics = mm.metrics.motchallenge_metrics
    metrics_host = mm.metrics.create()
    summary = Evaluator.get_summary(accumulators, sequences, metrics)
    summary_string = mm.io.render_summary(
        summary,
        formatters=metrics_host.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(summary_string)
    if args.save:
        current_time = datetime.datetime.now().strftime("%y%m%d-%H-%M-%S")
        summary_path = result_dir / f"mot_evaluation_{current_time}.csv"
        Evaluator.save_summary(summary, summary_path)
        print(f"Summary saved to {summary_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("result_dir")
    parser.add_argument("-s", "--save", action="store_true")
    main(parser.parse_args())
