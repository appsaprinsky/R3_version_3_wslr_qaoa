# Three-approach comparison on Amazon last-mile routes (Tuna-17)

| Approach (output folder) | What is solved how |
|---|---|
| `ORtoolsOutput` | the whole route, by OR-Tools CP-SAT (no start solution) |
| `AlgorithmWithORtoolCore` | my receding-horizon algorithm; every window solved by OR-Tools CP-SAT |
| `AlgorithmWithQC` | my receding-horizon algorithm; every window sampled on the Tuna-17 quantum computer |

Same routes, same cost function, one run per route and approach. Methodology and all decisions: `documentation/methodology.md`.

## 1. Install (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
qi login
```

Download the Amazon Last Mile Routing Research Challenge training data (https://routingchallenge.mit.edu/), unzip it into `data/`:

```bash
ls data/almrrc2021-data-training/model_build_inputs/
# route_data.json  travel_times.json  actual_sequences.json  ...
```

## 2. Prepare the routes

`config.toml` -> `[prepare]`: `num_routes`, `seed`, `exclude_route_ids`. There is no size filter on purpose.

```bash
python prepare_input.py
```

Saves the seeded random routes (matrix, Amazon sequence and cost) to `input_eval/`. To use more routes later, raise `num_routes`
and run it again: the sample is nested (existing routes stay, new ones are added).

## 3. Run

```bash
python main.py
```

Runs the approaches listed in `[experiment] approaches` one after another (never at the same time, so timings are not disturbed).
For `AlgorithmWithQC` it waits until the Tuna-17 jobs return. Stop it any time (Ctrl+C) and run the same command again to resume;
finished work is never recomputed. After raising `num_routes`, `python main.py` adds the new routes to every approach.

Dry run without a quantum computer: `backends = ["sim:Tuna-17"]` in `config.toml` (a local sampler, labelled SIMULATION).

## 4. Results

```
output_eval/
  ORtoolsOutput/results/             results.csv  anytime_trace.csv  summary.md  figures/
  AlgorithmWithORtoolCore/results/   results.csv  windows.csv  summary.md  figures/
  AlgorithmWithQC/results/           comparison.csv  windows.csv  jobs_ledger.csv  summary.md  figures/
  Comparison/                        paired_comparison.csv  statistics.md  statistics.json  figures/
```

```bash
cat output_eval/Comparison/statistics.md
```

Each approach folder also holds `runs/` (full state), `experiment.json` (settings, versions) and, for the QC approach, `backends.json`.
`AlgorithmWithQC/results/figures/subroutes/` shows every window: blue = order decoded from the QPU, red = part committed
(`visualise_subroute`).

## Check that real hardware was used

`main.py` prints `REAL QUANTUM HARDWARE` for Tuna-17 and refuses a backend the platform does not flag as hardware. To re-check a finished run:

```bash
python scripts/verify_hardware.py
```

## Start again

```bash
rm -rf output_eval/AlgorithmWithQC      # one approach only
rm -rf output_eval                    # everything (keeps input_eval/)
```

## Tests

```bash
python -m pytest tests -q
```

## If something goes wrong

| Message | Do |
|---|---|
| `Quantum Inspire is not logged in or the login expired. Run: qi login` | `qi login`, then `python main.py` |
| `The settings of <approach> changed since it was started` | restore the old `config.toml` values, or `rm -rf output_eval/<approach>` |
| `Route file(s) changed` | restore `input_eval/`, or `rm -rf output_eval/<approach>` |
| `CP-SAT did not prove the window optimal` | raise `[ortools_core] window_time_limit_s`; `python main.py` retries |
| waiting a long time on Tuna-17 | normal (shared queue); leave it running or Ctrl+C and rerun later |
