# Thesis Repository Release Checklist

## Code and models

- [ ] The final `app.py` uses the intended title and interface text.
- [ ] The application refuses to run without trained models.
- [ ] The application validates the model window duration.
- [ ] The final 60-second no-threshold model works with stability filtering disabled.
- [ ] The MET checkpoint loads successfully on CPU.
- [ ] LOSO and LOAO scripts display usable `--help` instructions.
- [ ] No source file contains a personal computer path.
- [ ] The final model artifacts are identified in `models/README.md`.

## Documentation

- [ ] Root `README.md` matches the final repository structure.
- [ ] Installation was tested in a clean virtual environment.
- [ ] `requirements-lock.txt` was generated from the validated environment.
- [ ] Input schemas match the final application.
- [ ] Reproduction commands were copied and tested.
- [ ] Thesis and publication citations are complete.
- [ ] Repository URL replaces any placeholder in the thesis.

## Privacy and permissions

- [ ] No raw participant data are present.
- [ ] `SubjectsInfo.xlsx` is absent.
- [ ] No signed consent or completed PAR-Q forms are present.
- [ ] Any IRB document is approved and appropriately redacted.
- [ ] No passwords, tokens, secrets, or private keys are present.
- [ ] Git history was reviewed for accidentally committed private files.
- [ ] Advisor approval was obtained for distributing the trained models.
- [ ] The repository license decision was confirmed.

## Verification

- [ ] A clean clone installs successfully.
- [ ] The Streamlit application starts.
- [ ] Approved example inputs complete the entire pipeline.
- [ ] The result uses the correct final model configuration.
- [ ] Aggregate reference metrics are documented.
- [ ] The commit hash used for the thesis submission is recorded.

## Formal release

- [ ] All final changes are committed.
- [ ] The repository is private unless public release was explicitly approved.
- [ ] A tag such as `thesis-v1.0` was created.
- [ ] A GitHub release was created from that tag.
- [ ] The release notes list the code, models, documentation, and known limitations.
- [ ] The advisor received the repository link, release tag, and access permission.
- [ ] The thesis states where the repository is maintained and which release corresponds to the submitted work.

## Recommended release record

Record the following in the thesis documentation or handoff package:

```text
Repository URL:
Release tag:
Commit hash:
Release date:
Access level:
Python version:
Primary contact:
```
