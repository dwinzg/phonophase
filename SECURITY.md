# Security

This is research code that reads audio files and CSV tables on the machine it
runs on. It runs no service and holds no credentials.

If you find a problem, for instance a crafted input file that makes a script
write outside its output path, report it privately through
[GitHub's private vulnerability reporting](https://github.com/dwinzg/phonophase/security/advisories/new)
and not in a public issue. Expect a reply within two weeks.

Only the current `main` branch is supported.

Two things to know when running it:

- Tables and manifests are plain CSV and are parsed, never executed. Model
  weights for the alignment pass come from the Hugging Face Hub; pin a revision
  if you need a fixed supply chain.
- The Colab notebook clones this public repository anonymously and needs no
  GitHub token.
