# Data Privacy and Repository Boundaries

## Material that may be stored in the repository

Subject to advisor and university approval, the repository may contain:

- Source code.
- Documentation.
- Aggregate performance tables.
- Figures that do not disclose participant information.
- Final trained models approved for distribution.
- Blank input templates.
- Links and citations to separately published datasets.
- A redacted IRB approval document if formal inclusion is authorized.
- A blank PAR-Q form if redistribution is permitted by its rights holder.

## Material that must not be committed

Do not add:

- Names, email addresses, phone numbers, or contact information.
- Signed informed-consent forms.
- Completed PAR-Q questionnaires.
- The private `SubjectsInfo.xlsx` workbook.
- Participant-level demographic records.
- Raw or processed recordings that are not explicitly approved for public release.
- Files connecting study identifiers to real identities.
- Credentials, passwords, tokens, private keys, or Streamlit secrets.
- Unredacted IRB correspondence containing signatures or personal contact information.

Pseudonymous participant numbers can still constitute research data. Participant-level result files should not be made public unless their release is covered by the approved protocol and confirmed by the advisor or responsible institutional office.

## Public dataset

The separately published multi-sensor dataset is available at <https://doi.org/10.5281/zenodo.15830858> under CC BY 4.0. Use the Zenodo record as the authoritative distribution source rather than duplicating the dataset in this repository.

When using the public dataset, retain its citation and license information. Do not assume that other study metadata or physiological targets are part of that public release.

## IRB and PAR-Q appendices

The thesis may contain the IRB approval and a blank PAR-Q instrument as appendices. Their presence in the thesis does not automatically authorize publication of completed forms, consent records, or additional study correspondence in the repository.

## Before publishing the repository

1. Keep the repository private during development.
2. Review the complete file list with the thesis advisor.
3. Search the repository history—not only the current files—for private data.
4. Remove secrets and private data from Git history if they were ever committed.
5. Confirm that trained-model distribution is permitted.
6. Confirm the selected software license or retain the no-license notice.
7. Release only the reviewed commit and tag.

If there is any uncertainty about whether a file can be shared, keep it outside the repository until written guidance is obtained.
