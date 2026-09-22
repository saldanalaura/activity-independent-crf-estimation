# Installation

## Supported environment

The repository is designed for Python on Windows, macOS, or Linux. The commands below use Windows PowerShell because that is the primary development environment. Python 3.10 or 3.11 is recommended for compatibility with the scientific and machine-learning dependencies.

Training the MET model can use a CUDA-capable GPU when PyTorch detects one. The Streamlit demonstration and XGBoost inference can run on a CPU.

## 1. Install prerequisites

Install:

- Python 3.10 or 3.11.
- Git.
- A current web browser for the Streamlit interface.

Confirm that Python is available:

```powershell
py --version
```

## 2. Obtain the repository

Using Git:

```powershell
git clone REPLACE_WITH_REPOSITORY_URL
cd VO2max_regression_thesis
```

If the repository is distributed as a ZIP file, extract it and open PowerShell in the extracted `VO2max_regression_thesis` directory.

## 3. Create an isolated environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the following once in the same terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```powershell
py -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository requires:

- Streamlit for the offline interface.
- NumPy, pandas, and SciPy for signal and feature processing.
- PyTorch for Stage 1 MET estimation.
- XGBoost for Stage 2 VO₂max estimation.
- scikit-learn for preprocessing and metrics.
- Matplotlib for saved LOAO figures.
- openpyxl for reading the subject-information workbook during experiment reproduction.

## 5. Verify the installation

```powershell
python -c "import numpy, pandas, scipy, sklearn, torch, xgboost, streamlit, matplotlib, openpyxl; print('Environment ready')"
```

Inspect the installed versions:

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import xgboost; print(xgboost.__version__)"
streamlit version
```

## 6. Start the demonstration

```powershell
streamlit run app.py
```

Streamlit normally opens `http://localhost:8501` automatically.

## 7. Record the release environment

Before the formal thesis release, create an exact dependency snapshot from the validated environment:

```powershell
pip freeze > requirements-lock.txt
```

Commit `requirements-lock.txt` with the final release. Keep `requirements.txt` as the readable list of direct dependencies and use the lock file to record exact versions.
