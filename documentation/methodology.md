# Methodology of the three-approach comparison

Purpose: a comparison that withstands reviewer scrutiny. Every decision that affects how results may be read is listed here.

## 1. Approaches

| | ORtoolsOutput | AlgorithmWithORtoolCore | AlgorithmWithQC |
|---|---|---|---|
| Solves | whole route | each window | each window |
| Solver | CP-SAT (`AddCircuit`, dummy end node) | the same CP-SAT model on the window | Tuna-17 (warm-start linear-ramp QAOA, `k^2` qubits) |
| Start solution | none | n/a | n/a |
| Stopping | `time_limit_s` per route, or proven optimum | all stops committed; every window must be proven optimal | all stops committed; fixed shots and COBYLA budget |
| Post-processing | none (raw) | closing 2-opt | closing 2-opt (same code) |

`AlgorithmWithORtoolCore` and `AlgorithmWithQC` share the code path for candidate selection (`k` nearest unvisited stops), commit depth `b`,
window matrix, commit and 2-opt. **The only difference is the window solver**, so a difference between them is attributable to the solver.
`k` and `b` are shared through `[qaoa]` (`k = "auto"` = 4 on Tuna-17, `b = k`); the CP-SAT core does not get a larger window.

## 2. Instances

* One seeded random sample of Amazon routes, **no size filter**, identical `input_eval/` files for all approaches. The sample is nested:
  the first N routes of the seeded order, so N can be raised later without changing earlier routes.
* The pilot route (used to choose settings) is excluded (`exclude_route_ids`), with a new seed.
* The cost is always `routing.open_cost` on the unrounded directed travel-time matrix: sum of consecutive edges, no return leg.
  Amazon's planned cost is a reference, not an approach.
* One run per route and approach; a route is the unit of the paired analysis.

## 3. Fairness decisions

1. **Baseline strength.** CP-SAT, 8 workers, 120 s, no hint. A start solution (nearest neighbour + 2-opt) changed the result by less than 1% on the pilot route (7,316 vs 7,340, optimum 7,278), so no hint is needed and no algorithm output leaks into the baseline.
2. **Equal post-processing.** The baseline is raw OR-Tools output. The algorithms include their designed 2-opt. Sensitivity columns: `cost_with_2opt_sensitivity` (baseline + 2-opt) and `*_cost_pre_2opt` (algorithms before 2-opt). In the pilot, 2-opt took the QC tour from 16,060 to 8,306, so the pre-2-opt cost is the cleaner view of the window solver.
3. **Equal budgets are read off, not imposed.** CP-SAT logs every improvement (solver wall time, CPU time, objective). `ortools_cost_at_*` in `paired_comparison.csv` is the OR-Tools cost reached within the time an algorithm run used. No OR-Tools re-run is needed.
4. **Threads.** CP-SAT uses `workers` threads, the algorithms one. Wall time and CPU time are both reported; do not compare wall time alone.
5. **Sequential execution.** Approaches never run at the same time, so CPU contention cannot disturb timings.
6. **Cost rounding.** CP-SAT optimises integer costs (x100 for the full route, x1000 for windows). Reported costs use the unrounded matrix; the rounding error is below 0.01 s per edge.
7. **No hidden restriction of the quantum approach.** No route-size filter; `k` is the hardware maximum (16 of 17 qubits); `p`, `epsilon`, shots and ramp settings are the intended design, fixed before the evaluation routes were drawn.
8. **Window-level pairing.** Every window the QC run visits is also solved by CP-SAT (diagnostic, excluded from QC timing); `windows.csv` reports the excess over that optimum. CP-SAT is checked against brute force on the same windows.

## 4. Runtime definitions (seconds)

* `compute_s` own computation: candidate selection, window preparation, angle optimisation on the exact simulator, circuit building, decoding, 2-opt.
* `subsolver_s` time inside CP-SAT (whole solve for ORtoolsOutput; sum over windows for the core). 0 for QC.
* `submit_s` QC: transpile + upload.  `qpu_exec_s` platform-reported execution time of the jobs (not documented to exclude overhead).
  `queue_wait_s` = turnaround (job created -> finished, platform timestamps) minus execution.
* `wall_excl_waiting_s` = compute + subsolver + submit + qpu_exec.  `wall_incl_waiting_s` = the same with turnaround instead of execution.
* `cpu_s` process CPU time over all threads (diagnostics excluded).

The QC compute time contains an exact simulation of the circuit for angle optimisation (exponential in `k^2`); a classical machine
can solve a window exactly in milliseconds. Runtime is therefore descriptive, not evidence of a quantum speed-up.

## 5. Analysis plan (fixed before the evaluation runs)

* Primary endpoint: gap to the best known cost per route, `100 (cost - best) / best`; best = lowest cost any approach (incl. sensitivity columns and the nearest-neighbour control) found on that route.
* Paired by route: median and quartiles of the gap, win / tie / loss counts, median paired difference, two-sided Wilcoxon signed-rank test. With few routes the test has little power; effect sizes and counts are primary.
* Pairs: QC vs core (solver effect), core vs ORtools (algorithm design), QC vs ORtools (overall), QC vs nearest neighbour (control), pre-2-opt costs for QC vs core.
* Failed QC runs are reported, never dropped. Hardware drift and shot noise are not captured by one run per route.

## 6. Known limitations

* The algorithm design (nearest-k candidate windows, `b = k`) may be weaker than plain nearest neighbour + 2-opt even with a perfect window solver (seen on the pilot route); `AlgorithmWithORtoolCore` shows this separately from any hardware effect.
* Under noise the QC decode returns the nearest-first order (Hungarian tie-break), so `AlgorithmWithQC` then behaves like nearest neighbour + 2-opt; the control and the window-level valid-shot fraction versus chance make this visible.
* QC queue time depends on platform load.
