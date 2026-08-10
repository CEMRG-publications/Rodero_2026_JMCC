# Example dataset

A small subset of the study data, included so that the repository can be cloned
and the analysis run immediately without downloading anything. It is enough to
check that an installation works and to see the drug-scenario analysis end to
end. It is not the full dataset: for that, see the Zenodo deposits listed in the
main README.

Run the analysis over it with:

```bash
./reproduce_figures.sh
```

## What is here

```
HCM/GSA_analysis/cycle/     Label dictionaries, parameter groupings, exclusions,
                            and the figure colour palettes used by the paper
HCM/<case>/scenarios/<scenario>/
├── data/
│   ├── xlabels.txt         46 input parameter names
│   └── ylabels.txt         48 simulation output names
└── output/
    ├── Si_total.csv        Total-order Sobol indices, parameters x outputs
    ├── Rank_Si_total_max_<output>.txt
    │                       Parameters ranked by their total-order index for
    │                       each of the 32 outputs analysed in the paper
    ├── GSA_wfrac_V_lower_0.0_upper_50.0/   mavacamten scenario, same files
    └── GSA_mu_V_lower_0.0_upper_50.0/      aficamten scenario, same files
```

The five cases are the five HCM anatomies of the paper:

| Case | Scenario | Phenotype |
|---|---|---|
| 1 | 53_more_samples | Mid-to-apical LVH |
| 2 | 47_more_samples | LVOTO |
| 3 | 48_more_samples | Isolated basal LVH |
| 4 | 49_more_samples | Milder asymmetric LVH |
| 5 | 50_more_samples | Undifferentiated pattern |

## Notes

The `xlabels.txt` and `ylabels.txt` files are byte-identical across the five
anatomies, so one copy is used for all of them.

The ranking files are the ones produced by the study, copied here rather than
regenerated. `Si_total.csv` is included as well, because some scripts recompute
the rankings from it through `generate_ranking_files.py`. The two files are not
interchangeable for every scenario: the `GSA_wfrac_V` matrices hold 32 columns,
one per analysed output, whereas the baseline and `GSA_mu_V` matrices hold all
48, so the column that a given output occupies differs between them.

This directory holds derived quantities only, no simulation output, no meshes,
and no patient data.
