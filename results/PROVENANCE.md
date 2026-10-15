# Where each table came from

`SHA256SUMS` fixes the bytes; `tests/test_results_integrity.py` checks them.
Every table is derived statistics only. No audio, see `NOTICE.md` for the
licence.

| file | produced by | checked |
|---|---|---|
| `eval_slices.csv.gz` | `scripts/slice_and_probe.py --hf-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA --limit-bonafide 1500 --limit-spoof 1500`, seed 1234, the command in `notebooks/confirmatory_run.ipynb`. 2,995 utterances, 137,992 rows. Both tables predate the `speaker` column, which analyses fill from the protocol file | 2026-10-14: rebuilt from the official eval FLACs with the tracked alignment cache, every row matches, largest difference 1.4e-9 |
| `dev_slices.csv.gz` | `scripts/slice_and_probe.py --subset dev`, seed 1234, local CPU. 5,092 utterances (all 2,548 bonafide), 228,362 rows. The exact spoof limit was not recorded | the same 5,092 utterances are reselected by `corpus.select_balanced(seed=1234)`, and the band edge calibration reproduces every row |
| `band_edge_alignment.csv.gz`, `band_edge_dev_alignment.csv.gz` | `scripts/band_edge.py align`: the aligner's phone timings per utterance, so slicing can be rerun without the model | used for both rebuilds above |
| `band_edge_descriptors.csv.gz`, `band_edge_dev_descriptors.csv.gz` | `scripts/band_edge.py describe` | `docs/BAND-EDGE.md` |
| `band_edge_summary.csv` | the `band_edge.py` measurements, one row per command, as quoted in `docs/BAND-EDGE.md` and `docs/PREREGISTRATION-BANDEDGE.md` | |
| `report_numbers.csv` | entered from the commands cited in its `source` column | `tests/test_report_numbers.py` checks each value lies in its interval and the headline row against the published value; the commands themselves are not rerun by the tests |

The aligner is `facebook/wav2vec2-lv-60-espeak-cv-ft`, pinned in
`scripts/slice_and_probe.py` to revision
`ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4` for new runs. The revision used
for the shipped tables was not recorded; their phone timings are kept in the
alignment files above, so nothing here depends on rerunning the model. The
Hugging Face mirror is pinned the same way, to revision
`e904f130e44a900cb7e444405d6b9edff30a5d19`, for new runs; the revision the eval
table was streamed from was not recorded, but the rebuild from the official
FLACs above shows the audio matches. Library versions that reproduce every number
are in `constraints.txt`.
