---
name: A number does not reproduce
about: A published number came out differently from its command
labels: reproducibility
---

**The number and where it is published** (file and line):

**The command you ran:**

**What it printed:**

**Environment:** Python version, `pip freeze | grep -E "numpy|scipy|librosa|scikit-learn|panphon"`, and whether you installed with `-c constraints.txt`.

**Checksum of the table you used** (`shasum -a 256 results/<table>`), to compare with `results/SHA256SUMS`:
