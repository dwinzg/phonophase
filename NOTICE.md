# Notice

The MIT licence in `LICENSE` covers the source code of this repository only.

## Data derived from ASVspoof 2019

The following files contain or are derived from the ASVspoof 2019 Logical
Access database:

| file | what it is |
|---|---|
| `results/eval_slices.csv.gz` | per phoneme slice statistics computed from the evaluation partition. No audio |
| `results/dev_slices.csv.gz` | the same statistics computed from the development partition. No audio |
| `configs/protocols/ASVspoof2019.LA.cm.eval.trl.txt` | the evaluation protocol file from the database, unchanged |

The ASVspoof 2019 database is made available under the Open Data Commons
Attribution License (ODC-BY) v1.0, https://opendatacommons.org/licenses/by/1-0/.
The files above are provided under the same terms, and any reuse of them must
keep this notice.

Database:

> Yamagishi, J., Todisco, M., Sahidullah, M., Delgado, H., Wang, X., Evans, N.,
> Kinnunen, T., Lee, K. A., Vestman, V. and Nautsch, A. (2019). ASVspoof 2019:
> The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge
> database. University of Edinburgh, The Centre for Speech Technology Research.
> https://doi.org/10.7488/ds/2555

Paper:

> Wang, X., Yamagishi, J., Todisco, M., Delgado, H., Nautsch, A., Evans, N.,
> et al. (2020). ASVspoof 2019: A large-scale public database of synthesized,
> converted and replayed speech. Computer Speech and Language, 64, 101114.
> https://doi.org/10.1016/j.csl.2020.101114

## Other corpora

No other corpus, and nothing derived from one, is distributed here. The terms of
every corpus this project reads are recorded per entry in
`configs/datasets.yaml`.
