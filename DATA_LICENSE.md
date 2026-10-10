# Data licence and provenance

## Input file

| Property | Value |
| --- | --- |
| File | `data/raw/Austin_TX_US_2020.csv` |
| Size | ~184 MB (193 MB reported by the blueprint) |
| Shape | 227,752 rows x 104 columns |
| Coverage | 13 smart-metered households, 30-minute interval, 1 Jan - 30 Dec 2020 |
| Nature | **Synthetic.** Uses Tunisian-grid-compatible electrical parameters (230 V, 50 Hz) on an Austin (Texas) calendar and climate. This framing must be stated in every document and in the paper. |
| Source | **To be confirmed** - see open question 1 below. |

## What is committed to the repository

The raw CSV exceeds GitHub's per-file size limit, so it is **not committed**.
`.gitignore` excludes `data/raw/*` except the manifest. What the repository does
contain:

- `data/raw/MANIFEST.csv` - SHA-256 checksum, size, and row/column counts, so
  any copy can be verified before a run.
- Derived artifacts regenerated from the CSV by code (`data/clean/`,
  `data/features/`, `data/injected/`), which are gitignored as Parquet.

## How to obtain the input

1. Obtain `Austin_TX_US_2020.csv` from the challenge organisers / the original
   `Data.zip` package (see open questions).
2. Place it at `data/raw/Austin_TX_US_2020.csv`.
3. Verify it: recompute the SHA-256 and compare it with `data/raw/MANIFEST.csv`.

## Open questions (must be resolved before submission)

- [ ] **Source and licence** of `Austin_TX_US_2020.csv` (generator, paper,
      Kaggle link) - determines what the repository may redistribute.
- [ ] Does the original `Data.zip` contain other cities or years? If so, they
      are the natural Phase 2 federated-continual-learning nodes.

Until the licence is confirmed, only the checksum manifest and code-derived
artifacts are published; the raw CSV stays local.
