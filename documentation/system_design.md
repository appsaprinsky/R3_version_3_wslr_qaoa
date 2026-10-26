# System design

> Updated for the three-approach experiment. The experimental protocol and every fairness decision are in `methodology.md`.
> `main.py` now runs `ORtoolsOutput`, `AlgorithmWithORtoolCore` and `AlgorithmWithQC` one after another, each with its own `Engine`,
> state folder (`output/<approach>/runs`), `experiment.json` (settings fingerprint + route hashes) and results; `Comparison/` reads them.
> New modules: `experiment.py` (reports + paired comparison). `ortools_solver.py` now also provides `solve_window` and an anytime trace.
> `prepare_input.py` no longer calls OR-Tools; it only prepares routes (seeded, nested, with an exclusion list).

Scope: one algorithm, **Warm-Start Linear-Ramp Receding-Horizon QAOA**, run on Quantum Inspire Tuna-5/9/17
over Amazon last-mile routes. Mathematics: `paper.pdf` (every formula there is what the code implements).
No quantum advantage is claimed.

## 1. Two commands

```
python prepare_input.py   config.toml [prepare]  ->  input/        (Amazon data -> prepared routes + baselines)
python main.py            input/ + config.toml   ->  output/        (QPU runs -> results)
```

## 2. Architecture

```
                         input/route_<id>.json  (cost matrix, Amazon sequence + cost, coords, OR-Tools)
                                     |
config.toml ----> main.py ----> Engine (engine.py)  ---- state: output/runs/<route>__<run>.json
                                     |                       (atomic writes after every transition)
          +--------------------------+----------------------------+
          | plan window              | solve window               | commit
   heuristic.py               window.py (+ qubo, warmstart,   heuristic.py
   (candidates, b, rho)        ramp, sim, circuit)            (tour, finalize: 2-opt)
                                     |
                              gateways.py
                    +----------------+-----------------+
          QuantumInspireGateway                    SimGateway (local, same interface)
          (QIProvider, backend.run,                (dry runs / tests)
           ResourceManager.get_job/get_results)
                                     |
                              report.py -> output/results/
```

| Module (`src/wslrqaoa/`) | Responsibility |
|---|---|
| `config.py` | load/validate `config.toml` (unknown keys rejected), backend table, window size `k = floor(sqrt(qubits))`, experiment fingerprint |
| `amazon.py` | read ALMRRC files, seeded route sampling, route record, sequence repair |
| `ortools_solver.py` | CP-SAT open-TSP baseline (dummy-node circuit) |
| `qubo.py` | window QUBO `Q`, penalty `P`, Ising coefficients |
| `warmstart.py` | LP relaxation (`surrogate` / `linearized`), clipping, `theta` |
| `sim.py` | exact numpy statevector simulator of the window circuit |
| `ramp.py` | linear-ramp angles; optimisation of `(delta_gamma, delta_beta)` |
| `circuit.py` | Qiskit circuit, calibration circuit |
| `window.py` | `make_window`, `choose_angles`, decoding and window diagnostics |
| `heuristic.py` | receding horizon: candidates, commit, 2-opt finalisation |
| `routing.py` | open-route cost (single function), asymmetric-correct 2-opt, exact window |
| `gateways.py` | Quantum Inspire and local-simulator gateways (same interface) |
| `engine.py` | runs, state machine, polling, retries, calibration, persistence |
| `report.py` | CSV / JSON / Markdown / figures |

`heuristic.py` and `routing.py` never import the gateways; the engine connects them.

## 3. Amazon dataset handling

Raw files (`data_dir/` or `data_dir/model_build_inputs/`): `route_data.json` (stops: lat, lng, type), `travel_times.json`
(directed seconds, stop x stop), `actual_sequences.json` (planned order, `"actual": {stop: position}`).

`prepare_input.py`:
1. Route ids = sorted keys of `travel_times.json` (optionally filtered by number of stops).
2. `numpy.random.default_rng(seed).permutation(ids)`; take routes in that order, skipping invalid ones
   (no `Station`, missing travel time, no planned sequence) until `num_routes` are found. Same seed + filter = same routes.
3. Per route: depot = first stop of type `Station`, re-indexed to node 0; dense cost matrix; planned sequence
   (sorted by position; repairs logged: duplicates removed, missing appended, depot rotated first);
   **`amazon_cost` = open-route cost of the planned sequence**; coordinates (plots only).
4. OR-Tools CP-SAT solution (`cost`, `status`, `lower_bound`), started from the cheaper of the Amazon route and nearest-neighbour + 2-opt (without a good start, CP-SAT is worse than NN + 2-opt on ~150-stop routes within minutes).
5. `input/route_<id>.json` + `input/manifest.json` (seed, filters, SHA-256 of the raw files).

`main.py` reads only `input/`. Cost function: `routing.open_cost` (no return leg), used everywhere.

## 4. Algorithm per window (see paper for proofs)

1. Candidates: `k` nearest unvisited (+ `rho*k` random), listed by distance.
2. QUBO (`k^2` qubits, index `i*k+t`), `P = lambda*k*max_edge` (guarantees feasible minimisers).
3. LP relaxation -> clip to `[eps, 1-eps]` -> `theta = 2 asin sqrt(x)`; state `RY(theta)|0>` per qubit.
4. Circuit: `p` x [cost layer `RZ`/`RZZ` from the Ising form, mixer `RY(theta) RZ(-2b) RY(-theta)`].
5. Ramp: `gamma_l = (l-1/2)/p * Dg`, `beta_l = (1-(l-1/2)/p) * Db`. `(Dg, Db)` minimise `<x^T Q x>/P`
   on the exact simulator (7x7 grid, then COBYLA) or are fixed from the config.
6. Sample on the QPU; decode `argmax_sigma sum_t m[sigma(t), t]` (Hungarian) from the shot marginals.
7. Commit the first `b` nodes; repeat. Finally 2-opt (exact deltas for asymmetric costs).

Runs per route: one `qpu` run per backend plus classical counterparts at the same `k`
(`ideal` = exact probabilities, `exact` = brute-force windows, `warm` = warm start only, `nn` = nearest neighbour).

## 5. Quantum Inspire handling

Reused calls from the reference scripts:
- submit: `QIProvider().backends()` -> backend by name -> `transpile(qc, backend, optimization_level=3, seed_transpiler=7)`
  -> `backend.run([circuits], shots=...)`; **never** `job.result()`.
- job ids: `job.circuits_run_data[i].job_id` (integers) + `job.batch_job_id`.
- retrieve: `ResourceManager().get_job(id).status` (`planned|running|completed|failed|cancelled`), then
  `ResourceManager().get_results(id)[0].results` = `{bitstring: count}` plus `shots_done`, `execution_time_in_seconds`, `created_on`.

Flow per window (state kept in `output/runs/*.json`):
```
prepare (QUBO, LP, angles, circuit) -> transpile + connectivity check -> submit -> SAVE job ids
   -> poll every poll_interval_s, no timeout -> all jobs done -> bit-order calibration (first window only)
   -> decode -> commit -> SAVE -> next window
```
- **Waiting:** the process stays in the polling loop. Status/network errors are logged and retried.
- **Failed/cancelled job:** resubmitted up to `max_job_retries` (new job id per attempt, old ids kept in `history`);
  then the run is `failed` and the others continue. `python main.py` retries failed runs.
- **Resume:** all state is on disk; exploration draws are seeded by (seed, run, iteration). A crash between the platform
  accepting a job and the ids being saved can duplicate one job (no corruption).
- **Calibration:** first batch per backend also contains `X` on qubit 0, all qubits measured. Qiskit order = qubit 0 is the *last* key character.
  Decision from single-bit frequencies (`p_last`, `p_first`): `p_last >= 0.6` and `p_last - p_first >= 0.4` -> decode as is; roles swapped -> reverse keys;
  otherwise the run stops, nothing is decoded. (The frequency of the one exact key falls with the qubit count: 46% on Tuna-17 in the first real run.)
- **Connectivity:** after transpiling, every two-qubit gate must act on a coupled pair and use a supported operation, else no submit.
- **Window size:** `k = floor(sqrt(qubits))`: Tuna-5 -> 2 (4 qubits), Tuna-9 -> 3 (9), Tuna-17 -> 4 (16). A `k` that does not fit is rejected.
- **Real hardware only:** at connect, `backend.get_backend_type().is_hardware` must be `True`, otherwise the program stops
  ("NOT confirmed real quantum hardware"). Emulators are reachable only through the explicit `sim:` prefix, which uses the local
  sampler and is labelled SIMULATION everywhere. After each job the platform's batch record (backend type + `is_hardware`,
  queued/reserved/finished times) is stored with the job. `python scripts/verify_hardware.py` re-checks all saved batches.
- **Recorded per job:** backend, p, angles, shots, transpiled depth / 2q gates / ops / qubit mapping, job and batch ids,
  submit / creation time, execution time, shots done, `error_mitigation: none`.

Sequential jobs per run = `ceil((n-1)/b)`; wall-clock time is dominated by queue waiting
(e.g. 150 stops: 75 jobs on Tuna-5, 50 on Tuna-9, 38 on Tuna-17 with `b = k`). Use `max_stops` for a short first run.

## 6. Timing

Per run (`output/runs/*.json` -> `timing`, summed over all steps, also across resumes):
`compute_s` = own computation (candidate selection, LP, angle optimisation on the exact simulator, circuit building, decoding, 2-opt),
`submit_s` = transpile + upload, `poll_s` = status/result calls. Per QPU job (platform data): `execution_time_in_seconds`
(Result, "time it took to compute the result"), job `created_on` / `queued_at` / `finished_at`, batch `queued_at` / `reserved_at` / `finished_at`.
Reported: `QPU exec` = sum of execution times; `turnaround` = job created -> finished (queue + execution);
`queue wait` = turnaround - exec; **incl. waiting** = compute + submit + turnaround; **excl. waiting** = compute + submit + exec.
`execution_time_in_seconds` is the platform's own figure; whether it contains control/compile overhead is not documented in the SDK.
OR-Tools: solver wall time. Amazon: planned by Amazon (no time).

## 7. Figures

`figures/routes/<route>/<path>__with_depot|without_depot.(png|pdf)`: one path per picture; coordinates are converted to km and the canvas
is sized to the exact bounds of what is drawn. `figures/subroutes/<route>/<run>/window_###.png` (`visualise_subroute = true`):
blue = full order decoded from the QPU for that window, red = the first `b` stops committed, black diamond = window start.

## 8. Files

`scripts/pilot_warmstart_bias.py` reproduces the pilot behind Remark 1 of the paper (ideal simulation on random windows, no Amazon data).

```
input/route_<id>.json, manifest.json
output/experiment.json            fingerprints of config + input, library versions
output/backends.json              topology, gate set, limits read from each backend
output/runs/<route>__<run>.json   full state of every run (tour, windows, jobs, counts)
output/results/                   summary.md, comparison.csv, windows.csv, jobs_ledger.csv, results.json, figures/
```
`main.py` refuses to resume if `config.toml` (algorithm settings) or `input/` changed (fingerprint check).

## 9. Not verified against the live service

Verified in the SDK source and with fake backends (Aer + restricted topologies): job-id semantics, status values,
`get_results` shape, transpile/connectivity guard, polling, retry, resume. **Not verifiable offline:** the exact
bit-string key format returned by Quantum Inspire (handled by the calibration probe), real queue behaviour, native gate sets of the Tuna backends,
and the platform's name strings (`Tuna-5/9/17`; matched case-insensitively, exact name first). First real run: use
`backends = ["Tuna-9"]` and `max_stops = 12` in `config.toml`.

## 10. Tests (`python -m pytest tests -q`)

QUBO = cost + penalty on random states and the all-zero state (regression for the diagonal bug); penalty proposition
(all minimisers valid) for k = 2..4; Ising mapping; LP surrogate by hand and linearised lower bound; circuit vs numpy vs Qiskit
`Statevector` (1e-10); mixer invariance; linear-ramp formula; decoding incl. reversed bit order; 2-opt exactness and local optimality;
OR-Tools vs brute force; seeded sampling and stop filter; sequence repair; end-to-end simulator runs; resume gives identical tours;
calibration (reversed / inconclusive); Quantum Inspire flow with fake provider (restricted topology, polling, retry, never-ready job, auth error).
