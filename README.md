# Data augmentation and external validation for 12-lead ECG classification (PTB-XL)

Code for the manuscript: simple augmentation vs a fairly tuned baseline on PTB-XL (five diagnostic superclasses),
subgroup tests, external validation on Chapman-Shaoxing/Ningbo, and a per-transformation ablation.

## Data (not included; download from PhysioNet)
- PTB-XL v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- Chapman-Shaoxing/Ningbo: https://physionet.org/content/ecg-arrhythmia/1.0.0/
Follow each dataset's licence and citation requirements.

## Environment
Python 3.12, a CUDA GPU (we used an RTX A4000, 16 GB). `pip install -r requirements.txt`, plus a CUDA build of torch.

## Run order (replace paths with yours)
```
python scripts/01_prepare_ptbxl.py <ptbxl_dir> ptbxl_out          # arrays + labels
python scripts/04_fair_baseline.py ptbxl_out 5 40                  # main comparison (baseline vs augmented, 5 seeds)
python scripts/03_experiments.py ptbxl_out 5 40                    # six-configuration ablation (regularisation, width, schedule)
python scripts/05_stats_and_figures.py ptbxl_out                   # statistics + figures 1-3
python scripts/06_final_tables.py ptbxl_out                        # validation-based selection + tables
python scripts/07_subgroup_tests.py ptbxl_out aug_constLR 2000     # sex/age tests (Holm-corrected)
python scripts/08_inspect_chapman.py <chapman_headers_dir> chapman_out   # parse headers + labels
python scripts/09_chapman_mapping_counts.py chapman_out            # class counts under the verified mapping
python scripts/10_train_save.py ptbxl_out 5 40                     # re-train and save weights for external validation
python scripts/11_external_chapman.py ptbxl_out chapman_out <chapman_mat_dir> 1000
python scripts/12_ablation.py ptbxl_out chapman_out 5 40           # per-transformation ablation
python scripts/13_fig_external.py                                  # external figure (values are typed in; edit to match your results)
```
`02_train_baseline.py` is the first single-run baseline and is kept for completeness.

## Notes
- Official PTB-XL folds: 1-8 train, 9 validation, 10 test. Normalisation statistics come from training folds only.
- External labels use only SNOMED codes named in the Chapman condition file; records with unverified frequent codes are
  excluded from negatives. MI could not be mapped. One damaged record is skipped.
- GPU runs are not bit-for-bit repeatable; expect differences in the third decimal.
- Set `random seeds` via the script arguments; results in the paper are 5-seed means.

## Licence
Add a licence (for example MIT) before publishing, and the paper citation once it is available.
