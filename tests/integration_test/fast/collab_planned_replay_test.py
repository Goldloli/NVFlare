# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a planned Collab async job through the real simulator.

``tests/unit_test/recipe/sim_env_test.py`` mocks ``job.simulator_run`` and
``tests/unit_test/collab/fedbuff_recipe_test.py`` injects a
``SimulationPlanRuntime`` into an in-process ``FLContext``, so the wiring
between them is untested: ``SimEnv`` writes the plan outside the job, the
simulator copies it to ``<workspace>/<job>/server/local/`` and passes
``--simulation-plan``, the server publishes the runtime on the job's
``FLContext``, and ``FedBuffWorkflow`` consumes it. These cases run that path
end to end with the NumPy example and read the trace under
``server/simulate_job/simulation/``.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

COLLAB_DIR = Path(__file__).resolve().parents[3] / "examples" / "advanced" / "collab"
JOB_NAME = "collab_numpy_async_reproducible"
PLANNED_ACCEPTANCE = [1, 0, 2, 3]
RUN_SEED = 10


def _load_example():
    """Import the reproducible NumPy example with its sibling modules resolvable."""
    sys.path.insert(0, str(COLLAB_DIR))
    try:
        return importlib.import_module("numpy_async_reproducible.job")
    finally:
        sys.path.remove(str(COLLAB_DIR))


def _run_planned_job(example, workspace_root, num_clients, plan_mode):
    from nvflare.recipe import SimEnv

    run = example.make_recipe(RUN_SEED).execute(
        SimEnv(
            num_clients=num_clients,
            workspace_root=str(workspace_root),
            reprod=True,
            plan=example.make_plan(RUN_SEED, plan_mode),
        )
    )
    assert run.get_status() == "FINISHED:COMPLETED", run.get_status()
    job_dir = workspace_root / JOB_NAME
    return {
        "job_dir": job_dir,
        "plan": job_dir / "server" / "local" / "simulation_plan.json",
        "trace": job_dir / "server" / "simulate_job" / "simulation",
    }


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _accepted(job):
    return _read_jsonl(job["trace"] / "accepted.jsonl")


def _assignments(job):
    return _read_jsonl(job["trace"] / "assignments.jsonl")


def _final_weights(job):
    return np.load(job["job_dir"] / "server" / "simulate_job" / "final_weights.npy")


@pytest.fixture(scope="module")
def fixed_runs(tmp_path_factory):
    """Run the fixed plan once per physical-worker count; the plan is the same."""
    example = _load_example()
    root = tmp_path_factory.mktemp("collab-planned-replay")
    return {count: _run_planned_job(example, root / f"workers-{count}", count, "fixed") for count in (1, 2)}


@pytest.fixture(scope="module")
def timed_run(tmp_path_factory):
    example = _load_example()
    return _run_planned_job(example, tmp_path_factory.mktemp("collab-timed"), 2, "timed")


@pytest.mark.timeout(300)
@pytest.mark.parametrize("num_clients", [1, 2])
def test_planned_replay_reaches_workflow(fixed_runs, num_clients):
    """The plan reaches the workflow with one and with two physical workers."""
    job = fixed_runs[num_clients]

    # The simulator kept the server-local copy, so this run cannot have fallen
    # back to live arrivals.
    execution = json.loads(job["plan"].read_text())["async_execution"]
    assert execution["arrival_sequence"] == PLANNED_ACCEPTANCE

    assignments = _assignments(job)
    accepted = _accepted(job)
    assert [item["assignment_id"] for item in accepted] == PLANNED_ACCEPTANCE
    assert [item["assignment_id"] for item in assignments] == [0, 1, 2, 3]
    assert [item["base_version"] for item in assignments] == [0, 0, 1, 2]
    assert [item["published_version"] for item in accepted] == [1, 2, 3, 4]

    # One physical worker executes both logical clients, in planned order.
    dispatch = _read_jsonl(job["trace"] / "dispatch.jsonl")
    expected_sites = {f"site-{number + 1}" for number in range(num_clients)}
    assert {item["site"] for item in dispatch} <= expected_sites
    assert [item["assignment_id"] for item in dispatch] == [0, 1, 2, 3]


@pytest.mark.timeout(300)
def test_planned_replay_is_independent_of_physical_worker_count(fixed_runs):
    """Bases, acceptance order and final weights match between worker counts."""
    one_client, two_clients = fixed_runs[1], fixed_runs[2]

    def recorded(job):
        return [(item["assignment_id"], item["base_version"], item["published_version"]) for item in _accepted(job)]

    assert (
        recorded(one_client)
        == recorded(two_clients)
        == [
            (1, 0, 1),
            (0, 0, 2),
            (2, 1, 3),
            (3, 2, 4),
        ]
    )
    np.testing.assert_array_equal(_final_weights(one_client), _final_weights(two_clients))


@pytest.mark.timeout(300)
def test_timed_plan_records_virtual_times(timed_run):
    """A timed plan replays on the virtual clock the plan declares."""
    timing = json.loads(timed_run["plan"].read_text())["async_execution"]["timing"]
    accepted = _accepted(timed_run)
    assert [item["assignment_id"] for item in accepted] == [0, 1, 2, 3]
    virtual_times = [item["virtual_time"] for item in accepted]
    assert all(value is not None for value in virtual_times)
    assert virtual_times == sorted(virtual_times)
    assert max(virtual_times) <= timing["horizon"]
