<div align="center">

# 🧬 StructuraMiner

### Exhaustive Structural Feature Extraction from Protein Data Bank (PDB) Files

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge)](https://github.com/<your-username>/StructuraMiner/releases)
[![Bioinformatics](https://img.shields.io/badge/Domain-Structural%20Bioinformatics-8A2BE2?style=for-the-badge)](https://github.com/<your-username>/StructuraMiner)

---

**StructuraMiner** is a production-grade, multi-level computational framework that exhaustively mines structural, physicochemical, topological, and interaction-based features from any Protein Data Bank (PDB) file — and outputs everything as clean, analysis-ready CSV tables.

[📖 User Manual (PDF)](docs/StructuraMiner_UserManual.pdf) · [🚀 Quick Start](#-quick-start) · [📦 Outputs](#-output-files) · [🔬 Applications](#-applications) · [💬 Issues](https://github.com/<your-username>/StructuraMiner/issues)

</div>

---

## 🖼️ Framework Overview

![StructuraMiner Workflow](docs/StructuraMiner_workflow.png)

*Complete multi-level extraction pipeline — from raw PDB input to seven analysis-ready output modules spanning atomic, residue, chain, and global levels.*

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **7 Output Modules** | Global · Per-Residue · Per-Atom · Per-Chain · Interactions · SS Segments · CA Distance Matrix |
| **~120 Features per Residue** | SASA, dihedrals, contacts, H-bonds, salt bridges, disulfides, π–π stacking, cation–π, network centrality, B-factors, and more |
| **Auto-Dependency Install** | Missing packages detected and installed automatically before any computation |
| **Dual PDB Format Support** | Full RCSB format (with HEADER/REMARK/SEQRES records) and stripped ATOM-only files |
| **Multi-Core Parallel** | Configurable CPU parallelism via `--cores` flag |
| **ML-Ready Output** | Tidy, long-format CSV files compatible with pandas, scikit-learn, PyTorch Geometric |
| **Zero Configuration** | Single script — no YAML, no config files, no project scaffold |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
gh repo clone lovekaushik899/StructuraMiner
cd StructuraMiner

# (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Self-healing mode:** If you skip the `pip install` step, StructuraMiner will detect and auto-install any missing packages on its first run.

### Basic Usage

```bash
# Single-core extraction (minimal)
python structuraminer.py --input my_protein.pdb

# Multi-core extraction with custom output prefix
python structuraminer.py --input my_protein.pdb --cores 8 --output my_run

# Quiet mode (suppress logging)
python structuraminer.py --input my_protein.pdb --cores 4 --quiet

# Check version
python structuraminer.py --version
```

### Python API

```python
from structuraminer import PDBFeatureFramework

framework = PDBFeatureFramework(
    pdb_path="my_protein.pdb",
    cores=4,
    output_prefix="my_run",
    verbose=True,
)

results = framework.run()

# Access individual dataframes
global_df      = results["global"]          # 1 row × N global features
per_res_df     = results["per_residue"]     # M residues × ~120 features
per_atom_df    = results["per_atom"]        # K atoms × atom-level features
per_chain_df   = results["per_chain"]       # C chains × chain features
interactions   = results["interactions"]   # All pairwise non-covalent contacts
ss_segments    = results["ss_segments"]    # HELIX / SHEET segments
dist_matrix    = results["distance_matrix"]  # CA-CA pairwise distances
```

---

## 📋 Command-Line Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *required* | Path to input `.pdb` file |
| `--cores` | `-c` | `1` | Number of CPU cores for parallel extraction |
| `--output` | `-o` | PDB file stem | Prefix for all output file names |
| `--quiet` | `-q` | `False` | Suppress verbose timestamped progress logging |
| `--version` | | | Print version number and exit |

---

## 📦 Output Files

StructuraMiner produces **8 output files** for every run, all prefixed with the input PDB stem (or `--output` prefix):

### 1. `{stem}_global_features.csv`
A single-row CSV capturing the protein's holistic properties.

| Feature Group | Key Columns |
|---|---|
| Metadata | `pdb_id`, `title`, `resolution_A`, `structure_method`, `space_group` |
| Composition | `n_residues`, `count_{AA}`, `frac_{AA}`, `n_chains`, `n_water`, `n_hetatm` |
| Physicochemistry | `mw_da`, `net_charge_pH7`, `pI_approx`, `frac_hydrophobic`, `mean_hydrophobicity_KD/Eisenberg` |
| Geometry | `radius_of_gyration_A`, `convex_hull_volume_A3`, `pca_eigenval_1-3`, `asphericity`, `prolateness` |
| B-factors | `bfactor_CA_mean/std/skew/kurt/IQR`, `bfactor_all_heavy_*` |
| Secondary Structure | `ss_frac_helix/sheet/coil`, `ss_n_helix/sheet_segments` |
| Crystallography | `unit_cell_a/b/c`, `unit_cell_alpha/beta/gamma` |

---

### 2. `{stem}_per_residue_features.csv`
The richest output: **~120 columns × N residue rows**. Powers residue-level ML models, Ramachandran analysis, and hotspot prediction.

<details>
<summary><b>Click to expand all feature groups</b></summary>

| Feature Group | Columns | Insight |
|---|---|---|
| Identity | `chain`, `resnum`, `resname`, `one_letter`, `residue_class` | Residue location and type |
| SASA | `sasa_total`, `sasa_backbone`, `sasa_sidechain`, `sasa_relative`, `is_buried`, `is_surface` | Solvent exposure; buried core vs. surface |
| Secondary Structure | `ss_code` (H/E/–) | DSSP-style assignment per residue |
| Backbone dihedrals | `phi_deg`, `psi_deg`, `omega_deg` | Ramachandran coordinates; cis-peptide detection |
| Side-chain dihedrals | `chi1_deg`, `chi2_deg` | Rotamer state; side-chain packing |
| Contacts (CA, 8 Å) | `n_contacts_CA`, `n_short/medium/long_contacts`, `mean_contact_dist` | Packing density |
| Contacts (heavy, 5 Å) | `n_contacts_heavy5` | Tight atomic packing |
| Hydrophobic contacts | `n_hydrophobic_contacts` | Hydrophobic core membership |
| Hydrogen bonds | `n_hbond_donor`, `n_hbond_acceptor`, `n_hbond_total` | H-bond network contribution |
| Salt bridges | `n_salt_bridges` | Electrostatic stabilisation |
| Disulfide bonds | `in_disulfide`, `ss_partner_chain`, `ss_partner_resnum` | Covalent cross-links |
| π–π stacking | `n_pipi_stacking` | Aromatic interactions |
| Cation–π | `n_cation_pi` | Electrostatic aromatic contacts |
| Backbone geometry | `bond_CA_N`, `bond_CA_C`, `bond_C_O`, `angle_N_CA_C`, `angle_CA_C_O`, `ca_displacement` | Structural quality; distortion detection |
| B-factors | `bfactor_res_mean/max/std`, `bfactor_backbone`, `bfactor_sidechain` | Residue flexibility |
| Network centrality | `network_degree`, `network_betweenness_centrality`, `network_closeness_centrality`, `network_eigenvector_centrality`, `network_clustering_coeff` | Structural & allosteric hubs |
| Water contacts | `n_water_contacts`, `min_water_dist_A` | Hydration shell |
| Neighbour composition | `nbr_frac_aliphatic`, `nbr_frac_aromatic`, `nbr_frac_polar_uncharged`, `nbr_frac_*charged` | Residue microenvironment |
| Physicochemical | `hydrophobicity_KD`, `hydrophobicity_Eisenberg`, `vdw_volume`, `residue_mw`, `formal_charge` | Biophysical properties |

</details>

---

### 3. `{stem}_per_atom_features.csv`
One row per heavy atom. Includes element, VdW radius, mass, B-factor, occupancy, and 3D coordinates. Ideal for atomic-resolution analyses and graph neural network atom-node featurisation.

---

### 4. `{stem}_per_chain_features.csv`
One row per polypeptide chain. Aggregated from per-residue data: length, composition, mean SASA, mean B-factor, secondary-structure fractions, net charge, and mean hydrophobicity.

---

### 5. `{stem}_interactions.csv`
Every detected pairwise non-covalent interaction. Columns: `interaction_type`, `chain1`, `resnum1`, `resname1`, `atom1`, `chain2`, `resnum2`, `resname2`, `atom2`, `distance_A`.

Supported interaction types:

| Type | Geometric Criterion |
|---|---|
| `hydrogen_bond` | N–O donor–acceptor distance ≤ 3.5 Å |
| `disulfide` | Cys SG–SG distance ≤ 2.2 Å |
| `salt_bridge` | Oppositely charged atom distance ≤ 4.0 Å |
| `pi_pi_stacking` | Ring centroid distance ≤ 7.0 Å; ring-plane angle ≤ 30° |
| `cation_pi` | Cationic atom within 6.0 Å of ring centroid |
| `hydrophobic_contact` | Hydrophobic residue CA–CA distance ≤ 5.0 Å |

---

### 6. `{stem}_ss_segments.csv`
One row per declared secondary structure element (HELIX or SHEET PDB record). Columns: `element_type`, `helix_id/sheet_id`, `chain`, `start`, `end`, `length`, `helix_type`, `sense`, `n_residues_in_segment`, `mean_bfactor`.

---

### 7. `{stem}_distance_matrix.csv`
Full upper-triangle CA–CA pairwise distance matrix in long format. Columns: `chain1`, `resnum1`, `resname1`, `chain2`, `resnum2`, `resname2`, `ca_dist_A`, `seq_separation`. Essential for contact-map visualisation and distance-map ML models.

---

### 8. `{stem}_extraction_summary.txt`
Human-readable provenance report: PDB metadata, composition summary, shape descriptors, secondary structure fractions, output file manifest, extraction time, and ISO timestamp.

---

## 📦 Dependencies

```
biopython>=1.79
numpy>=1.21
pandas>=1.3
scipy>=1.7
networkx>=2.6
freesasa>=2.1
pydssp>=0.8
tqdm>=4.60
```

Install via:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install biopython numpy pandas scipy networkx freesasa pydssp tqdm
```

> **Note:** `freesasa` provides Python bindings to the FreeSASA C library. On some platforms you may need to install system-level build tools (`gcc`, `make`) if a binary wheel is unavailable.

---

## 🔬 Applications

### 🤖 Machine Learning Feature Engineering
The per-residue CSV provides a ready-made feature matrix for residue-level classifiers (solvent exposure prediction, secondary structure assignment, binding-site detection). The global CSV is a molecular fingerprint for protein-level regressors (melting temperature, solubility, enzyme activity).

### 🧪 Structural Quality Assessment
B-factor distributions, Ramachandran outlier rates, and backbone bond-length deviations benchmark experimental structures or computationally predicted models (AlphaFold, RoseTTAFold) against high-resolution references.

### 🔁 Comparative Proteomics
Running StructuraMiner across a family of homologous structures generates aligned feature tables for evolutionary analysis: conservation in the buried core, secondary-structure content variation with thermostability, and more.

### 💊 Drug Discovery and Binding-Site Mapping
SITE records, interaction network topology, and SASA values combinedly pinpoint druggable pockets: surface-exposed cavities with local hydrophobicity, enriched in aromatic or charged residues, served by high-centrality hub residues.

### ⚙️ Protein Engineering and Stability Optimisation
Salt-bridge counts, disulfide-bond maps, packing ratios, and hydrophobic-contact density guide rational engineering campaigns. The distance matrix enables loop-modelling and linker-design decisions.

---

## 🧠 Biological Significance of Key Features

<details>
<summary><b>Solvent-Accessible Surface Area (SASA)</b></summary>

Computed by FreeSASA using the Lee–Richards rolling-probe algorithm. Relative SASA (rSASA) classifies each residue as:
- **Buried** (`rSASA < 0.25`): hydrophobic core, critical for stability
- **Surface-exposed** (`rSASA > 0.40`): drives interaction specificity

</details>

<details>
<summary><b>Backbone Dihedral Angles (φ, ψ, ω)</b></summary>

φ/ψ define Ramachandran space: allowed regions correspond to α-helices (~−60°, −45°), β-strands (~−120°, 120°), and left-handed helices. ω near 180° indicates trans-peptide; deviations flag rare cis-peptide bonds with functional significance (enzyme active sites, proline isomerisation).

</details>

<details>
<summary><b>Protein Contact Network Centrality</b></summary>

A CB–CB contact graph (< 8 Å) is constructed and five centrality metrics computed per residue:
- **Degree centrality**: number of direct contacts
- **Betweenness centrality**: residues acting as communication hubs
- **Closeness centrality**: proximity to all other residues
- **Eigenvector centrality**: influence in the global contact network
- **Clustering coefficient**: local network density

High-betweenness and high-eigenvector residues are structural/allosteric hubs — prime targets for mutagenesis and drug design.

</details>

<details>
<summary><b>Isoelectric Point (pI) Estimation</b></summary>

Computed via 150-iteration Henderson–Hasselbalch bisection accounting for all ionisable side chains (Asp, Glu, His, Cys, Tyr, Lys, Arg) and terminal charges. pI governs solubility, crystallisability, and interaction propensity at physiological pH.

</details>

<details>
<summary><b>Non-Covalent Interactions</b></summary>

The interactions CSV enumerates every detected non-covalent contact. Key contributions:
- **Hydrogen bonds**: stabilise secondary structure and enzyme active sites
- **Salt bridges**: contribute 1–5 kcal/mol per pair to thermostability  
- **π–π stacking**: major contributors to protein–ligand binding affinity
- **Cation–π interactions**: key in protein–protein and protein–DNA recognition

</details>

---

## ⚠️ Limitations and Notes

- **Multi-model PDB files** (e.g. NMR ensembles): only model 0 (first MODEL block) is processed. Split the file beforehand to analyse all models.
- **Hydrogen-bond detection** uses a simplified distance criterion (donor–acceptor ≤ 3.5 Å) rather than explicit H-atom placement. For high-precision H-bond enumeration, preprocess with Reduce or MolProbity.
- **pI estimation** uses standard solution-phase pKa values. Microenvironment shifts (buried titratable residues, metal coordination) are not modelled.
- **CA distance matrix** has O(N²) size. For large structures (> 2,000 residues), consider filtering by `seq_separation` or distance threshold for downstream use.
- **FreeSASA and pyDSSP** require well-formatted ATOM records. Heavily non-standard PDB files may produce `None` values in affected columns, which are preserved in the CSV.

---

## 📁 Repository Structure

```
StructuraMiner/
├── structuraminer.py                  # Main script — entire framework in one file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── README.md                    # This file
└── docs/
    ├── StructuraMiner_UserManual.pdf   # Full user manual (2–3 pages)
    └── StructuraMiner_workflow.png     # Publication-grade workflow diagram
```

---

## 📊 Example Output (excerpt)

**Per-residue features** (first 5 columns of many):

| chain | resnum | resname | ss_code | sasa_relative | phi_deg | psi_deg | n_contacts_CA | network_betweenness_centrality |
|---|---|---|---|---|---|---|---|---|
| A | 1 | MET | - | 0.81 | None | 152.3 | 4 | 0.0012 |
| A | 2 | ALA | H | 0.12 | -62.1 | -41.5 | 9 | 0.0034 |
| A | 3 | LEU | H | 0.08 | -58.7 | -43.2 | 11 | 0.0041 |
| A | 4 | GLU | H | 0.63 | -61.3 | -39.8 | 7 | 0.0019 |

**Global features** (subset):

| mw_da | pI_approx | radius_of_gyration_A | ss_frac_helix | ss_frac_sheet | frac_hydrophobic | net_charge_pH7 |
|---|---|---|---|---|---|---|
| 14307.2 | 6.84 | 11.23 | 0.412 | 0.183 | 0.342 | -2 |

---

## 🔖 Citation

If you use StructuraMiner in your research, please cite:

```bibtex
@software{StructuraMiner2024,
  author  = {<Your Name>},
  title   = {StructuraMiner: Exhaustive Structural Feature Extraction from PDB Files},
  version = {1.0.0},
  year    = {2024},
  url     = {https://github.com/<your-username>/StructuraMiner},
  license = {MIT}
}
```

---

## 🤝 Contributing

Contributions, feature requests, and bug reports are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/my-new-feature`
5. Open a Pull Request

Please open an issue first for major changes to discuss the proposed approach.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ for the structural bioinformatics community

**StructuraMiner v1.0.0** · [Report a Bug](https://github.com/<your-username>/StructuraMiner/issues) · [Request a Feature](https://github.com/<your-username>/StructuraMiner/issues)

</div>
