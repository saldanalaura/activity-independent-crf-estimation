# Activity-Independent Cardiorespiratory Fitness Estimation

This repository contains the code, trained-model artifacts, documentation, and offline demonstration developed for the master's thesis **Activity-Independent Estimation of Cardiorespiratory Fitness from Short-Duration Multimodal Wearable Signals** at the University of Puerto Rico at Mayagüez.

The project investigates whether cardiorespiratory fitness, represented by Queen's College Step Test (QCST)-derived VO₂max, can be estimated from short segments of multimodal wearable data without using activity identity as a predictor.

## Framework

The method has two modeling stages:

1. **Stage 1 — MET estimation.** Five-second inertial windows are converted into a continuous estimate of movement intensity expressed in metabolic equivalents (METs). The resulting sequence is smoothed and used to derive intensity-change and stability information.
2. **Stage 2 — VO₂max estimation.** Movement, heart rate, SpO₂, estimated MET, cross-modal, and participant features are combined into a 116-feature representation. An XGBoost regressor produces one VO₂max prediction per observation window, and the arithmetic mean of the available window predictions produces the participant-level estimate.

The experimental evaluation includes:

- Leave-One-Subject-Out (LOSO) validation.
- Leave-One-Activity-Out (LOAO) analysis.
- Observation windows of 2, 5, 10, 30, 60, and 90 seconds with 50% overlap.
- Configurations with and without MET-derived stability filtering.
- An offline Streamlit demonstration using previously recorded signals.

## Final thesis configuration

The final Stage 2 configuration uses:

- A 60-second observation window.
- 50% overlap.
- The complete 116-feature representation.
- No stability-based window exclusion.
- Participant-level aggregation by arithmetic mean.

The corresponding demonstration artifacts are:

```text
models/met_model_full
models/Window_60s/Without_threshold/final_all_subjects_model.json
models/Window_60s/Without_threshold/final_all_subjects_normalization.npz
```

## Repository structure

```text
VO2max_regression_thesis/
├── app.py
├── requirements.txt
├── README.md
├── CITATION.cff
├── LICENSE.md
├── pipeline/
│   ├── load_clean.py
│   ├── synchronize.py
│   ├── filtering.py
│   ├── windowing.py
│   ├── features_5s.py
│   ├── met_stage1.py
│   ├── features_60s.py
│   └── vo2_stage2.py
├── models/
│   ├── README.md
│   ├── met_model_full
│   └── Window_*/
├── training/
│   ├── README.md
│   ├── met_regression_loso_loao.py
│   ├── vo2max_regression_loso.py
│   └── vo2max_regression_loao.py
├── docs/
│   ├── installation.md
│   ├── user_guide.md
│   ├── data_format.md
│   ├── methodology.md
│   ├── reproducibility.md
│   ├── data_privacy.md
│   ├── troubleshooting.md
│   └── release_checklist.md
├── examples/
    └── README.md

```

## Quick installation

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
pip install -r requirements.txt
```

See [docs/installation.md](docs/installation.md) for complete instructions.

## Run the offline demonstration

```powershell
streamlit run app.py
```

Use the 60-second model without threshold listed under **Final thesis configuration**. Upload synchronized IMU and physiological CSV files, enter the participant characteristics, and run the pipeline. See [docs/user_guide.md](docs/user_guide.md) and [docs/data_format.md](docs/data_format.md) before using the interface.

## Reproduce the experiments

The training scripts receive every input and output path through command-line arguments; no path is tied to a particular computer.

```powershell
py training\met_regression_loso_loao.py --help
py training\vo2max_regression_loso.py --help
py training\vo2max_regression_loao.py --help
```

Complete commands and the required private input structure are documented in [docs/reproducibility.md](docs/reproducibility.md).

## Headline results

For the final 60-second, no-threshold configuration:

| Metric | Value |
|---|---:|
| Average participant-level RMSE | 5.4529 mL·kg⁻¹·min⁻¹ |
| Global RMSE | 6.7205 mL·kg⁻¹·min⁻¹ |
| R² | 0.4173 |
| Pearson correlation | 0.6498 |

Stage 1 obtained a LOSO RMSE of 0.885 MET with a bias of 0.018 MET. Refer to the thesis and associated manuscript for the complete analyses and interpretation.

## Data availability

Raw participant data and the subject-information spreadsheet are not stored in this repository. A related public multi-sensor dataset is available through Zenodo:

- J. L. Rivas-Caicedo, L. Saldaña-Aristizabal, K. Niño-Tejada, and J. F. Patarroyo-Montenegro, “A Multi-Sensor Dataset for Human Activity Recognition Using Inertial and Orientation Data,” *Data*, vol. 10, no. 8, article 129, 2025. <https://doi.org/10.3390/data10080129>
- Dataset: <https://doi.org/10.5281/zenodo.15830858>

The public dataset is distributed separately under CC BY 4.0. Its license does not automatically apply to the source code or trained models in this repository.

## Privacy and intended use

This software is a research prototype and is not a medical device. It must not be used for diagnosis, treatment decisions, emergency monitoring, or unsupervised health recommendations. Do not commit participant recordings, signed consent forms, completed PAR-Q questionnaires, or identifiable metadata. See [docs/data_privacy.md](docs/data_privacy.md).

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). Please cite the thesis and the associated publications when using this work.

## Funding

This research received support from the NSF EPSCoR Center for the Advancement of Wearable Technologies (CAWT), Grant No. OIA-1849243, and the NSF CAREER project “Intelligent Biomarker Analysis based on Wearable Distributed Computing,” Grant No. OAC-2439345.

## License and reuse

See [LICENSE.md](LICENSE.md). Unless another file explicitly states otherwise, no permission for redistribution or reuse is granted without written authorization from the copyright holder.
