import json
from pathlib import Path


class ExperimentStore:
    def __init__(self, output_dir="experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_results(self, results, filename="experiment_results.json"):
        output_file = self.output_dir / filename

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                results,
                file,
                indent=4
            )

        print(f"\nResults saved to: {output_file}")

    def load_results(self, filename="experiment_results.json"):
        output_file = self.output_dir / filename

        if not output_file.exists():
            return []

        with open(
            output_file,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)