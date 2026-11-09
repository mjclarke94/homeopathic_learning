"""Runner for this experiment: dry-run, list, run one cell, or emit one command per run.

    python experiments/number_sequences/german_dilution_qwen/run.py --dry-run
    python experiments/number_sequences/german_dilution_qwen/run.py --list
    python experiments/number_sequences/german_dilution_qwen/run.py --cell CELL_ID --seed SEED
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # run from a plain checkout; not needed after 'pip install -e .'
from hl.run import main  # noqa: E402

if __name__ == "__main__":
    main(default_config=Path(__file__).with_name("config.yaml"))
