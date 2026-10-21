# WS-LR-QAOA for the Open TSP on Quantum Inspire (Tuna-5 / Tuna-9 / Tuna-17)

Warm-Start Linear-Ramp Receding-Horizon QAOA on Amazon last-mile routes.
Design and maths: `documentation/system_design.md`, `documentation/paper.pdf`.

## 1. Install (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
qi login
```


If login fails, try
```bash
python -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
```
Download the Amazon Last Mile Routing Research Challenge training data
(https://routingchallenge.mit.edu/), unzip it into `data/`, and check that this exists:

```bash
ls data/almrrc2021-data-training/model_build_inputs/
# route_data.json  travel_times.json  actual_sequences.json  ...
```

## 2. Prepare the input (once per sample)

Set the sample in `config.toml` under `[prepare]` (`num_routes`, `seed`, optional `min_stops` / `max_stops`), then:

```bash
python prepare_input.py
```

This selects random Amazon routes (seeded, reproducible), saves them with their actual Amazon cost
and an OR-Tools solution to `input/`.

## 3. Run

```bash
python main.py
```

`main.py` reads `input/`, submits the circuits to Tuna-5, Tuna-9 and Tuna-17, **waits until the jobs
return** (polling, no timeout), advances the algorithm, and writes `output/results/`.
It can be stopped at any time (Ctrl+C, closed laptop). Run the same command again to resume.

Defaults: 1 route, 1 run per QPU, `p = 2`, 1024 shots. All parameters are in `config.toml`.

## 4. Results

```bash
cat output/results/summary.md
```

| File (`output/results/`) | Content |
|---|---|
| `summary.md` | Amazon vs OR-Tools vs Tuna-5/9/17, classical counterparts, window diagnostics |
| `comparison.csv` | one row per route (all costs, improvements, gaps) |
| `windows.csv` | one row per window (valid-shot fraction, TVD, regret, job ids, transpiled depth) |
| `jobs_ledger.csv` | every Quantum Inspire job id |
| `figures/` | cost bars, route maps (with / without depot), scaling plot (PNG + PDF) |

## Dry run without a QPU (no account needed)

In `config.toml` set `backends = ["sim:Tuna-9"]`, delete `output/`, then `python main.py`.

## Start a new experiment

```bash
rm -rf output          # keeps input/
# or, for a new route sample:
rm -rf input output && python prepare_input.py
```

## Tests

```bash
python -m pytest tests -q
```

## If something goes wrong

| Message | Do |
|---|---|
| `Quantum Inspire is not logged in or the login expired. Run: qi login` | `qi login`, then `python main.py` |
| `config.toml or input/ changed since this experiment was started` | restore the old settings, or `rm -rf output` |
| `ATTENTION ... failed` | `python main.py` retries failed runs; see `output/runs/*.json` |
| waiting a long time | normal (shared queue); leave it running or Ctrl+C and rerun later |
