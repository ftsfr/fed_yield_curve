import os
import platform
import subprocess
import sys
from pathlib import Path

import chartbook
from doit.tools import run_once

sys.path.insert(1, "./src/")

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"
OUTPUT_DIR = BASE_DIR / "_output"
OS_TYPE = "nix" if platform.system() != "Windows" else "windows"

# The core pipeline needs no credentials. Setting WRDS_USERNAME in .env
# additionally enables the CRSP Treasury pull and the full GSW figure
# replication notebook (see README).
WRDS_USERNAME = chartbook.env.get("WRDS_USERNAME", default="")
HAS_WRDS = WRDS_USERNAME not in ("", "your_wrds_username")


os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"


# Paths are double-quoted (not single-quoted) so they work with spaces on
# both sh and Windows cmd.exe, which does not treat ' as a quote character.
# fmt: off
def jupyter_execute_notebook(notebook_path):
    return f'jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace "{notebook_path}"'
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f'jupyter nbconvert --to html --output-dir="{output_dir}" "{notebook_path}"'
# fmt: on


def mv(from_path, to_path):
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f'mv "{from_path}" "{to_path}"'
    else:
        command = f'move "{from_path}" "{to_path}"'
    return command


def task_config():
    """Create empty directories for data and output if they don't exist"""
    def create_dirs():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "actions": [create_dirs],
        "targets": [DATA_DIR, OUTPUT_DIR],
        # Directory targets can't be fingerprinted, so without this the task
        # would rerun on every invocation; it still reruns if the dirs vanish.
        "uptodate": [run_once],
        "verbosity": 2,
    }


def task_pull():
    """Pull Federal Reserve yield curve data"""
    yield {
        "name": "fed_yield_curve",
        "actions": [
            f"python ./src/pull_fed_yield_curve.py",
        ],
        "targets": [
            DATA_DIR / "fed_yield_curve.parquet",
            DATA_DIR / "fed_yield_curve_all.parquet",
        ],
        "file_dep": [
            f"./src/pull_fed_yield_curve.py",
        ],
        "clean": [],
    }
    if HAS_WRDS:
        yield {
            "name": "crsp_treasury",
            "actions": ["python ./src/pull_CRSP_treasury.py"],
            "targets": [
                DATA_DIR / "CRSP_TFZ_INFO.parquet",
                DATA_DIR / "CRSP_TFZ_DAILY.parquet",
            ],
            "file_dep": ["./src/pull_CRSP_treasury.py"],
            "clean": [],
        }
    else:
        # Keep the subtask discoverable without WRDS credentials: it succeeds
        # with a note (printed on the first run only, to keep repeated `doit`
        # runs no-ops) instead of failing or disappearing from `doit list`.
        def explain_crsp_pull():
            print(
                "Skipped: CRSP Treasury pull requires a WRDS account.\n"
                "To enable it, copy .env.example to .env, set WRDS_USERNAME,\n"
                "and rerun `doit`. This adds the CRSP-based reproductions of\n"
                "GSW (2007) Figures 1, 2, and 6 to the replication notebook."
            )
        yield {
            "name": "crsp_treasury",
            "actions": [explain_crsp_pull],
            "uptodate": [run_once],
            "verbosity": 2,
        }


def task_format():
    """Format data into standardized FTSFR datasets"""
    return {
        "actions": [
            f"python ./src/create_ftsfr_datasets.py",
        ],
        "targets": [
            DATA_DIR / "ftsfr_treas_yield_curve_zero_coupon.parquet",
        ],
        "file_dep": [
            f"./src/create_ftsfr_datasets.py",
            DATA_DIR / "fed_yield_curve.parquet",
        ],
        "clean": [],
    }


TEST_RESULTS = DATA_DIR / "test_results.txt"


def run_tests_and_record(
    results_file=TEST_RESULTS, test_file="./src/test_gsw_svensson.py"
):
    """Run pytest and write its output to results_file only if the suite passes.

    The results file is the doit target for the tests task: it is never written
    on failure (and any stale copy is removed), so a failed run leaves the task
    out of date and doit reruns it, while a passing run keeps `doit` idempotent.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v"],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        Path(results_file).unlink(missing_ok=True)
        return False
    Path(results_file).write_text(result.stdout)
    return True


def task_tests():
    """Run the pytest suite (GSW verification against the pulled data)."""
    return {
        "actions": [run_tests_and_record],
        "file_dep": [
            "./src/test_gsw_svensson.py",
            "./src/gsw_pricing.py",
            "./src/pull_fed_yield_curve.py",
            DATA_DIR / "fed_yield_curve_all.parquet",
            *(
                [
                    DATA_DIR / "CRSP_TFZ_INFO.parquet",
                    DATA_DIR / "CRSP_TFZ_DAILY.parquet",
                ]
                if HAS_WRDS
                else []
            ),
        ],
        "targets": [TEST_RESULTS],
        "task_dep": ["pull:fed_yield_curve"],
        "verbosity": 2,
    }


notebook_tasks = {
    "summary_fed_yield_curve_ipynb": {
        "path": "./src/summary_fed_yield_curve_ipynb.py",
        "file_dep": [
            DATA_DIR / "ftsfr_treas_yield_curve_zero_coupon.parquet",
            DATA_DIR / "fed_yield_curve.parquet",
            DATA_DIR / "fed_yield_curve_all.parquet",
        ],
        "targets": [],
    },
    # Always runs; without WRDS credentials the CRSP-based figures are
    # replaced by a note explaining how to enable them.
    "gsw_crsp_replication_ipynb": {
        "path": "./src/gsw_crsp_replication_ipynb.py",
        "file_dep": [
            DATA_DIR / "fed_yield_curve_all.parquet",
            *(
                [
                    DATA_DIR / "CRSP_TFZ_INFO.parquet",
                    DATA_DIR / "CRSP_TFZ_DAILY.parquet",
                ]
                if HAS_WRDS
                else []
            ),
        ],
        "targets": [],
    },
}
notebook_files = []
for notebook in notebook_tasks.keys():
    pyfile_path = Path(notebook_tasks[notebook]["path"])
    notebook_files.append(pyfile_path)


# fmt: off
def task_run_notebooks():
    for notebook in notebook_tasks.keys():
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix(".ipynb")
        yield {
            "name": notebook,
            "actions": [
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
            ],
            "task_dep": ["tests"],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on


def task_generate_charts():
    """Generate interactive HTML charts."""
    return {
        "actions": ["python src/generate_chart.py"],
        "file_dep": [
            "src/generate_chart.py",
            DATA_DIR / "ftsfr_treas_yield_curve_zero_coupon.parquet",
        ],
        "targets": [
            OUTPUT_DIR / "yield_curve_replication.html",
        ],
        "verbosity": 2,
        "task_dep": ["format", "tests"],
    }


def task_generate_pipeline_site():
    return {
        "actions": [
            "chartbook docs generate -f",
        ],
        "targets": ["docs/index.html"],
        "file_dep": ["chartbook.toml", OUTPUT_DIR / "yield_curve_replication.html", *notebook_files],
        "task_dep": ["generate_charts"],
    }
