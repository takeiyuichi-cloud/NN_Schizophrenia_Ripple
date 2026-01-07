Repository Overview

This repository contains the analysis code, derived datasets, and group-level outputs supporting the findings reported in the accompanying manuscript on hippocampal–cortical ripple dynamics in schizophrenia.
The primary goal of this repository is to enable transparent inspection, full computational reproduction, and extension of the reported analyses, while respecting data privacy and ethical constraints associated with human neurophysiological recordings.

All analyses were performed on anonymized, preprocessed data.
No raw MEG recordings or personally identifiable information are included.
The repository focuses on subject-level summaries, surrogate-based metrics, and group-level representations that are sufficient to reproduce all figures, tables, and statistical results reported in the manuscript.

Importantly, all statistical results and figures were recomputed from the included data tables, rather than being post hoc summaries of previously reported statistics.

⸻

Scope of the repository

This repository supports the following key components of the manuscript:

- Detection and characterization of temporally clustered sharp-wave ripples (SWRs) across hippocampal and cortical sources  
- Group comparisons between healthy controls (HC) and individuals with schizophrenia (SZ)  
- Surrogate-based validation of temporal clustering  
- Analysis of hippocampo–cortical transition dynamics  
- Associations between ripple metrics, large-scale coordination, and clinical symptom severity  
- Generation of all main figures (Figs. 1–5) and all supplementary tables

The repository is designed to be analysis-complete, meaning that all numerical results, figures, and statistical inferences in the manuscript can be regenerated using the provided scripts and data.

⸻

Repository structure

```markdown
.
├── data/
│   ├── subject-level and group-level analysis tables
│   ├── surrogate-aware clustering outputs
│   ├── anonymized clinical association tables
│   └── group-averaged neuroimaging maps (NIfTI)
│
├── scripts/
│   ├── figure-specific analysis scripts (Figs. 1–5)
│   ├── supplementary table builders
│   └── utility scripts for data aggregation and validation
│
├── outputs/
│   ├── generated figures (PDF)
│   └── generated tables (CSV/XLSX)
│
├── run_all.py
├── README.md
└── LICENSE
``` 

Data organization

The data/ directory contains only derived or anonymized datasets, including:
	•	Subject-level summary tables (e.g., ripple counts, transition shares)
	•	Surrogate-based clustering metrics
	•	Group-level source maps used for visualization
	•	Files containing subject-to-anonymous-ID mappings only when required for reproducibility

No raw MEG recordings, sensor-level data, or identifiable metadata are included.

A detailed Data Dictionary describing each file is provided below in this README.

⸻

Code organization
	•	Each major figure in the manuscript is associated with one or more dedicated analysis scripts located in the scripts/ directory.
	•	Scripts are named to reflect their corresponding figure or supplementary table
(e.g., fig3_bde_with_supp_table5_to7.py).
	•	All scripts are written in Python and rely on standard scientific libraries
(NumPy, pandas, SciPy, statsmodels, matplotlib).

Where relevant, scripts explicitly document:
	•	Statistical models used (e.g., permutation tests, GLMs, GEE)
	•	Covariates and multiple-comparison correction procedures
	•	Design choices inherited from earlier analysis stages

⸻

Reproducibility and transparency

One-command reproduction

All figures and supplementary tables can be regenerated using a single entry-point script:

```markdown
python run_all.py
```

This script sequentially executes all figure- and table-specific analysis scripts in the correct order, regenerating every numerical result and figure reported in the manuscript.

Optional execution modes

•	Dry run (no execution, print pipeline only):

```markdown
  python run_all.py --dry-run
```

•	Run only selected figures:

```markdown
  python run_all.py --only fig3 fig5
```

•	Continue despite errors (not recommended for formal reproduction):

```markdown
  python run_all.py --continue-on-error
```

All outputs are written to the outputs/ directory, and a timestamped log file is generated for full traceability.

Reproducibility assumptions
	•	Scripts are executed from the repository root.
	•	A standard Python scientific environment is available.
	•	Randomized procedures (e.g., permutation tests, bootstrap confidence intervals) use fixed random seeds where applicable.

While exact floating-point values may differ slightly across platforms, all qualitative results, statistical significance patterns, and conclusions are reproducible.

⸻

Ethical considerations and data privacy

All data included in this repository comply with institutional review board (IRB) approvals and data-sharing agreements.
	•	Only derived, anonymized, and non-identifiable data are shared.
	•	The repository does not allow reconstruction of individual raw recordings.
	•	Files labeled *_id_map_private.csv contain only internally consistent anonymized identifiers required for linkage across derived tables and cannot be used to re-identify individuals.

⸻

Data Dictionary

This repository provides anonymized, analysis-ready datasets and derived group-level outputs supporting the figures and statistical analyses reported in the manuscript.
All files listed below are used directly by the analysis scripts included in this repository.

⸻

Core analysis tables

transition_share_master.csv

Master table of subject-level transition shares (%) across all hippocampo–cortical transition types. Used as the primary input for transition analyses (Figs. 4–6).

transition_share_counts_subject_level.csv

Subject-level transition counts (n) and denominators (total) for each transition type. Used for binomial GLM / GEE analyses of transition shares.

rate_epoch_subject_level.csv

Subject-level summary of high-rate ripple epoch metrics (e.g., number of epochs, event counts, durations) for each frequency band. Used for Fig. 3 and Supplementary Tables S5–S7.

rate_epoch_epoch_level.csv

Epoch-level table containing individual high-rate ripple epochs and their durations. Used to compute mean epoch duration statistics.

high_rate_epoch_debug_merged.csv

Surrogate-aware debug table containing observed and surrogate clustering metrics. Required for surrogate-excess analyses (Supplementary Table S6).

⸻

Clinical association source tables

fig5a_source_public.csv

Anonymized subject-level table containing pooled ripple counts, demographic covariates, and clinical measures. Used for Fig. 5a and related supplementary analyses.

fig5b_source_public.csv

Anonymized subject-level table used for analyses of clustered ripple dynamics and clinical associations (Fig. 5b, Supplementary Table S11).

source_fig5_predicted_share_by_load_symptom_tertiles_d.csv

Derived analysis table used to generate predicted transition-share curves stratified by symptom severity and ripple load.

⸻

Network- and spectral-level summaries

events_by_network_subject_level.csv

Subject-level counts of ripple events aggregated by functional network.

df_PSD.csv

Power spectral density (PSD) summaries used for spectral analyses and normalization steps.

cooc_global_by_window_allpooled.csv

Global co-occurrence matrix of ripple events across sliding windows, pooled across subjects.

df_clean_expanded.csv

Cleaned and expanded subject-level table containing merged metadata and derived variables used across multiple analyses.

⸻

Transition-related auxiliary data

hippo_cortex_count_ratio_by_subject_pooled_80_240.csv

Subject-level pooled hippocampal and cortical ripple counts (80–240 Hz), used to compute total ripple load and z-scored load covariates.

IEI_pairs_subject_medians_0to500ms.csv

Subject-level median inter-event intervals (IEIs, 0–500 ms) for each transition pair. Used for Fig. 4b.

⸻

Group-level source maps (Fig. 2a)

The fig2a/ directory contains group-averaged source-space maps used to generate Figure 2a.

These files are provided in NIfTI format (.nii.gz) and represent spatial distributions of ripple-related activity separately for healthy controls (HC) and individuals with schizophrenia (SZ), across multiple ripple frequency bands.
All maps are group-level averages and do not contain individual subject data.

File naming convention

```markdown
<Region>_<Group>_<Frequency>.nii.gz

•	Region
 	•	Hippocampus – hippocampal source space
 	•	Cortex – cortical source space
•	Group
 	•	HC – healthy controls
 	•	SZ – schizophrenia
•	Frequency
  •	Ripple frequency band (80, 120, 160, 200, or 240 Hz)
```

Files included
	•	80 Hz
	•	Hippocampus_HC_80Hz.nii.gz
	•	Hippocampus_SZ_80Hz.nii.gz
	•	Cortex_HC_80Hz.nii.gz
	•	Cortex_SZ_80Hz.nii.gz
	•	120 Hz
	•	Hippocampus_HC_120Hz.nii.gz
	•	Hippocampus_SZ_120Hz.nii.gz
	•	Cortex_HC_120Hz.nii.gz
	•	Cortex_SZ_120Hz.nii.gz
	•	160 Hz
	•	Hippocampus_HC_160Hz.nii.gz
	•	Hippocampus_SZ_160Hz.nii.gz
	•	Cortex_HC_160Hz.nii.gz
	•	Cortex_SZ_160Hz.nii.gz
	•	200 Hz
	•	Hippocampus_HC_200Hz.nii.gz
	•	Hippocampus_SZ_200Hz.nii.gz
	•	Cortex_HC_200Hz.nii.gz
	•	Cortex_SZ_200Hz.nii.gz
	•	240 Hz
	•	Hippocampus_HC_240Hz.nii.gz
	•	Hippocampus_SZ_240Hz.nii.gz
	•	Cortex_HC_240Hz.nii.gz
	•	Cortex_SZ_240Hz.nii.gz

Description

Each NIfTI file contains a spatial map reflecting the mean ripple-related source activity within the specified region, frequency band, and diagnostic group.
These maps were generated by averaging subject-level source estimates after spatial normalization and are intended to support:
	•	Reproduction of Fig. 2a
	•	Independent visualization or re-thresholding
	•	Comparison with external neuroimaging datasets or atlases

⸻

Sample / demonstration data

sample_candidate_120Hz.npz

Small example file illustrating candidate ripple events at 120 Hz. Provided for demonstration and testing purposes only.

⸻

Notes on data access and privacy
	•	All subject identifiers are anonymized.
	•	No raw MEG data are included.
	•	Files labeled *_id_map_private.csv are required for internal linkage during analysis but do not contain direct personal identifiers.
  These mapping files contain only internally consistent anonymized identifiers and cannot be used to re-identify individuals.
  
