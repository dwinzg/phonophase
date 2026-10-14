# Where each table came from

`SHA256SUMS` fixes the bytes; `tests/test_results_integrity.py` checks them.
Every table is derived statistics only. No audio, see `NOTICE.md` for the
licence.

| file | produced by | checked |
|---|---|---|
| `eval_slices.csv.gz` | `scripts/slice_and_probe.py --hf-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA --limit-bonafide 1500 --limit-spoof 1500`, seed 1234, the command in `notebooks/confirmatory_run.ipynb`. 2,995 utterances, 137,992 rows. It predates the `speaker` column, which analyses fill from the protocol file | 2026-10-14: rebuilt from the official eval FLACs with the tracked alignment cache, every row matches, largest difference 1.4e-9 |
| `dev_slices.csv.gz` | `scripts/slice_and_probe.py --subset dev`, seed 1234, local CPU. 5,092 utterances (all 2,548 bonafide), 228,362 rows. The exact spoof limit was not recorded | the same 5,092 utterances are reselected by `corpus.select_balanced(seed=1234)`, and the band edge calibration reproduces every row |
| `band_edge_alignment.csv.gz`, `band_edge_dev_alignment.csv.gz` | `scripts/band_edge.py align`: the aligner's phone timings per utterance, so slicing can be rerun without the model | used for both rebuilds above |
| `band_edge_descriptors.csv.gz`, `band_edge_dev_descriptors.csv.gz` | `scripts/band_edge.py describe` | `docs/BAND-EDGE.md` |
| `band_edge_summary.csv` | the `band_edge.py` measurements, one row per command, as quoted in `docs/BAND-EDGE.md` and `docs/PREREGISTRATION-BANDEDGE.md` | |
| `report_numbers.csv` | entered from the commands cited in its `source` column | `tests/test_report_numbers.py` checks each value lies in its interval and the headline row against the table |

The aligner is `facebook/wav2vec2-lv-60-espeak-cv-ft` at revision
`ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`, pinned in
`scripts/slice_and_probe.py`. Library versions that reproduce every number
are in `constraints.txt`.
