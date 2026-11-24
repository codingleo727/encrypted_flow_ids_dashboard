# Encrypted Flow IDS — tuned CPU beta, v0.2

Offline binary flow classification using **XGBoost, a PyTorch 1D-CNN, and a
weighted ensemble**. This update adds the agreed hyperparameter search,
full-sampled-training refits, ensemble/calibration selection and validation-tuned
F1 thresholds. All selection uses 2017 development data. The held-out 2017 test
and 2018 external evaluation are reserved for final reporting.

**Start with [QUICKSTART.txt](QUICKSTART.txt).** No original CIC datasets,
fitted research models or measured benchmark performance are bundled.

## Setup and execution

Use 64-bit Python 3.11 or 3.12 in a virtual environment. No GPU or NPU is required.
On Windows, after creating `.venv`, substitute `.\.venv\Scripts\python.exe` for
`python` in these examples. On Linux/macOS use `.venv/bin/python`.

```bash
python -m pip install -r requirements.txt
python -m ids doctor
python -m unittest discover -s tests -v
python -m ids inspect --data data/raw/2017
python -m ids train --data2017 data/raw/2017 --run runs/tuned --config configs/tuning.json
python -m ids evaluate --data2018 data/raw/2018 --run runs/tuned --max-rows 150000
python -m streamlit run dashboard.py
```

Extract labeled flow CSVs into `data/raw/2017/` and `data/raw/2018/`. Recursive
subfolders are supported. Keep years separate and exclude prediction outputs
or unrelated CSVs. The exact saved feature schema is required for prediction.
Compatible original 2017 long-form and 2018 abbreviated headers are mapped
explicitly; missing features and unrecognized labels fail rather than silently
changing the task. Review semantics/units if using a modified data release.

The precise HP laptop hardware has not been verified. The default CPU thread
request is four, capped at the available logical CPU count. Runtime is data and
hardware dependent; reported timing estimates use completed trials on your
machine. They estimate the remaining search for that model only and exclude
later model searches and final refits. Budget several hours or an overnight run.
No bit-for-bit reproducibility claim is made across OS, libraries or hardware.
Seeds and deterministic PyTorch algorithms are enabled for repeatable runs in
the same environment. Record exact versions with `python -m pip freeze`.

For Linux CPU-only installs, installing PyTorch from its CPU index before
`requirements.txt` avoids unnecessary CUDA packages. For Windows native-library
errors consult the [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html)
(including Visual C++ Redistributable) and the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
Python and installed wheels must support the same CPU architecture.

## Data boundaries and preprocessing

1. **Bounded initial sample.** Scan every supplied 2017 CSV in chunks and retain
   the K rows with the smallest seeded random priorities. This is a uniform,
   label-independent sample, not an initial-file prefix. `tuning.json` and
   `laptop.json` use K=150,000; `quick.json` uses K=40,000. These are pre-cleaning
   caps, not the number of final training examples. CSV I/O still scans all rows.
2. **Numeric cleaning.** Replace nonnumeric, infinite, negative and float32-overflow
   feature values with missing. Drop rows with no numeric candidate features.
   The selected statistics are nonnegative sizes, timings, rates and counts.
   Embedded CSV header rows are removed; missing/unknown labels raise errors.
3. **Duplicate control.** Hash the candidate float32 feature vectors. Remove
   contradictory labels for the same vector and keep one copy of exact duplicate
   vectors before splitting. Metadata never enters the classifiers.
4. **2017 split.** Approximately 60% training / 20% validation / 20% test. If all
   source CSVs have complete IP/port/protocol metadata, group by an undirected
   connection five-tuple; no connection group crosses these boundaries. Otherwise
   use binary-stratified random splitting after exact-vector deduplication.
   Related host/capture/session patterns may still cross partitions. This is not
   evidence of temporal independence or deployment-level generalization.
5. **Three validation roles.** Within the validation partition, approximately
   50% is for search/early stopping, 25% for calibration fits, and 25% for blend
   and F1-threshold selection. Where metadata exists these roles also preserve
   connection groups. Very small class counts or insufficient groups cause a
   clear error; increase the initial sample cap. Actual counts and assignments
   are exported. Search and early stopping necessarily reuse the search subset,
   and weight and threshold selection reuse the decision subset. These are
   development results, not independent performance estimates.
6. **Train-only transforms.** Candidate columns use the existing related-statistic
   order (up to 55). IPs, ports, flow IDs, timestamps and source file names are
   excluded. Ports remain excluded; no port ablation is performed because the
   agreed target is cross-dataset transfer. Drop all-missing or zero-variance
   columns based only on the full sampled training partition. XGBoost uses raw
   unscaled values with native missing-value handling. The CNN uses training-only
   median imputation and sklearn RobustScaler. Robust-scaled inputs are clipped
   to +/-100 IQR units as a numerical safeguard for extreme CIC outliers. The
   feature mask, medians, centers and scales are saved without pickle.

`features` in the manifest is the complete required input schema; `active_features`
is the subset actually used by both models. Demo arrays and overlap signatures
retain the full candidate schema. Constant-column decisions never use test data.

## Hyperparameter search and final refits

The search is seeded random sampling of unique combinations, with bounded
validation-diagnostic proposals inside the same trial budget. It does not need
Optuna. First draw a **tuning training subset**: preserve every training example
of Heartbleed, Infiltration, Web Attack/XSS/SQL Injection, Bot and brute-force
families; retain 25% (rounded up, minimum one) of each other original label.
Validation keeps its natural retained distribution and is not downsampled.

**Rare preservation applies only to examples already in the sampled training
partition.** Initial uniform sampling can omit rare source labels. The program
logs low-support warnings and exports `label_support.csv`, including zero counts
for source labels missing from a partition. A large aggregate score does not
establish reliable detection of every rare attack.

| Model | Trials | Search parameters |
| --- | ---: | --- |
| XGBoost | 18 | class weight: 1, sqrt(negative/positive), ratio; column fraction: .5/.7/.9; depth: 4/6/8; child weight: 1/3/10 |
| 1D-CNN | 14 | learning rate: .0003/.001/.003; dropout: .2/.35/.5; class weight: 1/sqrt-ratio/ratio; kernel: 3/5/7 |

XGBoost fixed settings: CPU histogram trees, up to 1,500 trees, learning rate
.05, subsample .85, column fraction per level .8, gamma .1, L1=.5, L2=2. Early
stopping explicitly **maximizes validation PR-AUC**, patience 30, saving the best
iteration rather than a stack including later stopped trees.

CNN: three Conv1d + BatchNorm + ReLU blocks, channels 32/64/128, same padding,
adaptive average pooling to length one, dense 64 + ReLU + dropout + dense one.
BCEWithLogitsLoss uses the selected positive weight. AdamW weight decay=.001,
batch size=512, up to 35 epochs. ReduceLROnPlateau maximizes validation PR-AUC
(factor .5, patience 3); early stopping patience=6. Shuffling is seeded and uses
zero worker processes. Batches need not contain positives; binary loss still
handles them. There is no claim of guaranteed per-batch class balance.

Both weighting formulas are recomputed from the **actual partition being fit**:
tuning subset counts during search, full sampled training counts during final
refits. Each winning configuration is refit once using all sampled training
rows, with the search-validation subset still supplying early stopping. It never
combines validation/test rows into the final training partition.

### PR metric definition

The common selection objective is **trapezoidal PR-AUC**, calculated with
`sklearn.metrics.precision_recall_curve` and `auc`. XGBoost uses a custom callback
metric with explicit `maximize=True`, so it has exactly the same definition as
the CNN and ensemble. It is not XGBoost's built-in `aucpr` calculation and is not
sklearn **average precision**. Average precision is exported separately. Ties and
curve interpolation can make the two PR summaries differ; report their names
accurately rather than comparing unlike metrics.

### Diagnostic fallbacks and runtime

Diagnostic thresholds are investigation heuristics, not guaranteed requirements.
At the provisional .5 threshold, low rare recall (support at least five) proposes
increased class weight and XGBoost child weight=1; excessive FPR/low precision
proposes a lower class weight. CNN collapse proposes sqrt-ratio weighting and
learning rate at most .001. A large XGBoost validation PR-AUC jump (>.1 between
rounds) proposes learning rate=.03 and max_delta_step=2. Duplicate effective
configurations are avoided. These proposals consume existing trial slots; they
do not launch unbounded extra searches. Final F1 thresholds are chosen afterward.

If a tuning CNN epoch exceeds `cnn_epoch_budget_seconds` (default 180), that
trial becomes ineligible and counts toward the trial budget. Remaining trials
use channels 16/32/64 and batch size 1024. If no eligible trial completes, training
fails clearly. The final full-training refit retains the winning architecture;
its epoch time limit is not enforced because it has more training rows. Lower
sample size or configured epochs/trials before starting if compute is limited.

Trials and best search checkpoints are saved incrementally. An interrupted run
is retained for inspection; automatic resume is not implemented. Choose a fresh
run name after correcting a failure. Existing nonempty runs are never overwritten.

## Ensemble and classification thresholds

After final refits, calculate predictions on the 2017 calibration and decision
subsets. Evaluate `w * XGBoost + (1-w) * CNN` for w=0,.1,...,1 on the decision
subset and select the highest PR-AUC; exact ties prefer weights closer to .5.
Endpoints are valid: the selected ensemble can reduce to one base model.

If the best raw blend does not improve on the best base PR-AUC, fit regularized
sigmoid/Platt-style calibration on each model's **calibration subset** logit
scores, then evaluate the calibrated blend grid on the separate decision subset.
Select calibration only if it improves decision PR-AUC over the raw candidate.
Base-model metrics retain raw model outputs; the ensemble may apply its saved
calibrators. Neither raw nor calibrated scores are guaranteed to be accurate
probabilities, especially on the 2018 distribution.

Select a separate binary attack F1-maximizing threshold for XGBoost, CNN and the
chosen ensemble using the decision subset only. Exact F1 ties choose the highest
threshold to flag fewer flows. Freeze and save all three thresholds, weight and
optional calibrators in `decisions.json`; reuse them everywhere without refitting.
FPR is reported, not forced below 2%. Calibration and ensemble selection add no
further deep-model training.

## Final evaluation and demo

Only after all decisions are frozen, evaluate the three models on the 2017 test
partition. External evaluation is a separate command with no training/tuning
operation. It takes a bounded uniform sample of the supplied 2018 CSVs, removes
internal duplicate/conflicting vectors and exact matches to any retained 2017
vector, then applies the saved schema, preprocessing, models and decisions.
It fails on incompatible/missing features. Do not change configurations based on
2018 results and still call that dataset an unseen final test.

Report precision, recall, binary F1, macro F1, weighted F1, ROC AUC, trapezoidal
PR-AUC, average precision, FPR and per-original-label detection rates/support.
Undefined recall/FPR/AUC is null when a class is absent. The rare-attack flag in
`attack_breakdown.csv` identifies the specified families. FPR=FP/(FP+TN).

Timing is median warmed CPU **inference including model-specific preprocessing,
calibration and blending**, up to 10,000 rows in batches of 1,024, repeated three
times. Ensemble latency includes both model calls. It excludes CSV I/O and model
loading and is not single-record response latency. Original beta runs retain
their previously recorded model-only timings; do not compare scopes as equivalent.

The dashboard supports all three models and uses saved thresholds, actual
measured metrics, per-label supports, trial tables, CNN history, feature
importance, and ensemble settings. Clicking Run detection predicts a cached
sample of up to 2,000 evaluation flows; sample F1 is separate from full saved
sample metrics. Ground-truth labels enter metrics after prediction only.

The system uses payload-free flow statistics. Dataset membership in an
encrypted-only subset has not been established. The CNN learns across aggregate
feature positions, not packet sequences or temporal packet ordering. This is
an offline demonstration, with no live capture or traffic-blocking function.

## Commands and artifacts

```bash
# Short real-data timing experiment, still using only 2017:
python -m ids train --data2017 data/raw/2017 --run runs/timing --config configs/quick.json --xgb-trials 1 --cnn-trials 1
# Streaming inference, labels optional, saved thresholds applied:
python -m ids predict --data data/raw/2018 --run runs/tuned --model all --output predictions_2018.csv
```

`train` overrides: `--max-rows`, `--epochs`, `--threads`, `--xgb-trials`,
`--cnn-trials`. Additional settings are in the JSON presets. Unknown keys or
invalid values fail. `configs/smoke.json` is strictly a small software test
(2+2 trials), not the agreed experiment budget.

| File | Purpose |
| --- | --- |
| `manifest.json` | Completion status, schema v2, active features, configuration and environment |
| `preprocess.npz` | Training-fitted feature mask and CNN transforms |
| `xgboost.json`, `cnn.pt` | Final full-sampled-training winners |
| `trials.json`, `trials.csv` | Incremental candidate settings, status, scores, diagnostics, timings |
| `search_best_xgboost.json`, `search_best_cnn.pt` | Best tuning-stage checkpoints (not the final models) |
| `best_parameters.json`, `training.json` | Selected settings and final refit details |
| `decisions.json` | Ensemble grid, calibration selection, thresholds and provenance |
| `label_support.csv` | Source, sample, training, validation-role and test supports, including zeros |
| `data2017.json`, `split.json`, `splits.csv.gz` | Source inventory, sampling, cleaning and exact partition assignments |
| `cnn_history.csv`, `feature_importance.csv` | Final CNN history and XGBoost interpretation |
| `2017_validation/` | Development metrics; includes reused development subsets, not an independent test |
| `2017_test/`, `2018_unseen/` | Metrics JSON/CSV, compressed predictions, attack breakdown |
| `2018_unseen/data_report.json` | External sampling and overlap-removal counts |
| `reference_hashes.npy` | Retained 2017 vector signatures |
| `demo_2017.npz`, `demo_2018.npz` | Raw feature/label/metadata samples for the demo |

Copy the completed run folder and project source to move the demo. The raw
original CSVs are unnecessary at demo time. Keep raw data, virtual environments
and unrelated artifacts out of a deployment package. The original `.gitignore`
excludes run folders: explicitly include the intended trained run if deploying
via Git. Hosted deployment remains a separate step.

## Software verification

```bash
python -m unittest discover -s tests -v
```

Core tests cover schema alignment, cleaning, provenance, duplicate filtering,
class mapping, deterministic sampling, split boundaries, train-only transforms,
search candidate uniqueness, rare-family subsampling, validation-role isolation,
PR definitions, class-weight formulas, F1 thresholds and ensemble/calibration
math. Integration tests require all model/UI dependencies and use tiny synthetic
fixtures to exercise actual search/refits, save/load agreement, overlap removal,
unchanged external-evaluation artifacts, label-free CSV prediction and dashboard
controls for all three models. Synthetic results are not research performance.

For a synthetic demonstration, follow the smoke commands in QUICKSTART and use
`runs/synthetic` in the sidebar. The app displays a synthetic-data warning.

## Sources

- [CIC-IDS2017, CIC / UNB](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CSE-CIC-IDS2018, CSE and CIC / UNB](https://www.unb.ca/cic/datasets/ids-2018.html)
  and [AWS registry](https://registry.opendata.aws/cse-cic-ids2018/)
- Sharafaldin, Lashkari and Ghorbani (2018), *Toward Generating a New Intrusion
  Detection Dataset and Intrusion Traffic Characterization*, ICISSP. Follow the
  original dataset citation requirements in your submission.
- [XGBoost early stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html),
  [PyTorch](https://pytorch.org/docs/stable/index.html),
  [sklearn PR metrics](https://scikit-learn.org/stable/modules/model_evaluation.html),
  [Streamlit](https://docs.streamlit.io/)
