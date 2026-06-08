#!/usr/bin/env python3

# ─── DEPENDENCY BOOTSTRAPPER ─────────────────────────────────────────────────
import sys
import subprocess

REQUIRED = {
    "biopython":  "Bio",
    "numpy":      "numpy",
    "pandas":     "pandas",
    "scipy":      "scipy",
    "networkx":   "networkx",
    "freesasa":   "freesasa",
    "pydssp":     "pydssp",
    "tqdm":       "tqdm",
}

def _install_missing():
    """Auto-install any missing packages before importing them."""
    missing = []
    for pkg, imp in REQUIRED.items():
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[SETUP] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--break-system-packages"] + missing,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("[SETUP] Installation complete.\n")

_install_missing()

# ─── STANDARD IMPORTS ────────────────────────────────────────────────────────
import argparse
import os
import re
import math
import time
import warnings
import traceback
import multiprocessing
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import cdist, squareform
from scipy.stats import skew, kurtosis
from tqdm import tqdm

from Bio import PDB
from Bio.PDB import (
    PDBParser, PPBuilder, NeighborSearch,
    calc_dihedral, calc_angle, is_aa,
    Selection, vectors
)
from Bio.PDB.vectors import Vector

import freesasa
from pydssp import assign as dssp_assign, read_pdbtext as dssp_read

# ─── CONSTANTS & LOOKUP TABLES ───────────────────────────────────────────────

VERSION = "1.0.0"

# Kyte-Doolittle hydrophobicity scale
KD_HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9,  'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
}

# Eisenberg consensus hydrophobicity
EISENBERG_HYDROPHOBICITY = {
    'ALA': 0.620, 'ARG': -2.530, 'ASN': -0.780, 'ASP': -0.900, 'CYS': 0.290,
    'GLN': -0.850, 'GLU': -0.740, 'GLY': 0.480, 'HIS': -0.400, 'ILE': 1.380,
    'LEU': 1.060, 'LYS': -1.500, 'MET': 0.640, 'PHE': 1.190, 'PRO': 0.120,
    'SER': -0.180, 'THR': -0.050, 'TRP': 0.810, 'TYR': 0.260, 'VAL': 1.080,
}

# Residue van der Waals volumes (Å³) - Richards 1974
VDW_VOLUME = {
    'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5,
    'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1,  'HIS': 153.2, 'ILE': 166.7,
    'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7,
    'SER': 89.0,  'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0,
}

# Residue molecular weights (Da)
RESIDUE_MW = {
    'ALA': 89.09, 'ARG': 174.20, 'ASN': 132.12, 'ASP': 133.10, 'CYS': 121.16,
    'GLN': 146.15, 'GLU': 147.13, 'GLY': 75.03,  'HIS': 155.16, 'ILE': 131.17,
    'LEU': 131.17, 'LYS': 146.19, 'MET': 149.21, 'PHE': 165.19, 'PRO': 115.13,
    'SER': 105.09, 'THR': 119.12, 'TRP': 204.23, 'TYR': 181.19, 'VAL': 117.15,
}

# Formal charge at pH 7 (integer)
FORMAL_CHARGE = {
    'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
    'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0,  'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0,  'PRO': 0,
    'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0,  'VAL': 0,
}

# Amino acid pKa side chain values
SIDE_CHAIN_PKA = {
    'ASP': 3.65, 'GLU': 4.25, 'HIS': 6.00,
    'CYS': 8.18, 'TYR': 10.07, 'LYS': 10.53, 'ARG': 12.48,
}

# Residue classification
RESIDUE_CLASS = {
    'ALA': 'aliphatic_nonpolar', 'VAL': 'aliphatic_nonpolar',
    'ILE': 'aliphatic_nonpolar', 'LEU': 'aliphatic_nonpolar',
    'MET': 'aliphatic_nonpolar', 'PRO': 'aliphatic_nonpolar',
    'PHE': 'aromatic',           'TRP': 'aromatic',
    'TYR': 'aromatic',           'GLY': 'special',
    'CYS': 'special',            'SER': 'polar_uncharged',
    'THR': 'polar_uncharged',    'ASN': 'polar_uncharged',
    'GLN': 'polar_uncharged',    'ASP': 'negatively_charged',
    'GLU': 'negatively_charged', 'LYS': 'positively_charged',
    'ARG': 'positively_charged', 'HIS': 'positively_charged',
}

# Backbone reference max SASA (Å²) for relative SASA calculation (Wilkinson 1991)
MAX_SASA = {
    'ALA': 121.0, 'ARG': 265.0, 'ASN': 187.0, 'ASP': 163.0, 'CYS': 148.0,
    'GLN': 214.0, 'GLU': 194.0, 'GLY': 97.0,  'HIS': 216.0, 'ILE': 195.0,
    'LEU': 191.0, 'LYS': 230.0, 'MET': 203.0, 'PHE': 228.0, 'PRO': 154.0,
    'SER': 143.0, 'THR': 163.0, 'TRP': 264.0, 'TYR': 255.0, 'VAL': 165.0,
}

# One-letter codes
ONE_LETTER = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

# Van der Waals radii for atoms (Å)
ATOM_VDW_RADIUS = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
    'S': 1.80, 'P': 1.80, 'F': 1.47, 'CL': 1.75,
    'BR': 1.85, 'I': 1.98, 'FE': 1.80, 'ZN': 1.39,
    'CA': 2.31, 'MG': 1.73, 'MN': 1.80, 'CU': 1.40,
}

# Rotamer chi1 definitions (atom names for chi1 dihedral)
CHI1_ATOMS = {
    'ARG': ('N','CA','CB','CG'), 'ASN': ('N','CA','CB','CG'),
    'ASP': ('N','CA','CB','CG'), 'CYS': ('N','CA','CB','SG'),
    'GLN': ('N','CA','CB','CG'), 'GLU': ('N','CA','CB','CG'),
    'HIS': ('N','CA','CB','CG'), 'ILE': ('N','CA','CB','CG1'),
    'LEU': ('N','CA','CB','CG'), 'LYS': ('N','CA','CB','CG'),
    'MET': ('N','CA','CB','CG'), 'PHE': ('N','CA','CB','CG'),
    'PRO': ('N','CA','CB','CG'), 'SER': ('N','CA','CB','OG'),
    'THR': ('N','CA','CB','OG1'),'TRP': ('N','CA','CB','CG'),
    'TYR': ('N','CA','CB','CG'), 'VAL': ('N','CA','CB','CG1'),
}

# Chi2 definitions
CHI2_ATOMS = {
    'ARG': ('CA','CB','CG','CD'), 'ASN': ('CA','CB','CG','OD1'),
    'ASP': ('CA','CB','CG','OD1'), 'GLN': ('CA','CB','CG','CD'),
    'GLU': ('CA','CB','CG','CD'), 'HIS': ('CA','CB','CG','ND1'),
    'ILE': ('CA','CB','CG1','CD1'), 'LEU': ('CA','CB','CG','CD1'),
    'LYS': ('CA','CB','CG','CD'), 'MET': ('CA','CB','CG','SD'),
    'PHE': ('CA','CB','CG','CD1'), 'PRO': ('CA','CB','CG','CD'),
    'TRP': ('CA','CB','CG','CD1'), 'TYR': ('CA','CB','CG','CD1'),
}

# Disulfide bond length threshold (Å)
SSBOND_DIST = 2.2

# Hydrogen bond geometric criteria
HBOND_DA_DIST   = 3.5   # Donor–Acceptor distance (Å)
HBOND_ANGLE_MIN = 120.0 # Minimum D-H···A angle (degrees)

# Salt bridge distance threshold
SALTBRIDGE_DIST = 4.0   # Å

# Contact distance thresholds
CA_CONTACT_DIST = 8.0   # Å
CB_CONTACT_DIST = 8.0   # Å
HEAVY_CONTACT_DIST = 5.0  # Å

# Hydrophobic contact threshold
HYDROPHOBIC_DIST = 5.0  # Å

HYDROPHOBIC_RES = {'ALA','VAL','ILE','LEU','MET','PHE','TRP','PRO','TYR'}
AROMATIC_RES    = {'PHE','TRP','TYR','HIS'}
POLAR_POS_RES   = {'ARG','LYS','HIS'}
POLAR_NEG_RES   = {'ASP','GLU'}

# PI-PI stacking thresholds
PIPI_DIST  = 7.0  # Å centroid–centroid
PIPI_ANGLE = 30.0 # degrees deviation from parallel

# Aromatic ring centroids (atom names)
RING_ATOMS = {
    'PHE': ['CG','CD1','CD2','CE1','CE2','CZ'],
    'TYR': ['CG','CD1','CD2','CE1','CE2','CZ'],
    'TRP': ['CG','CD1','CD2','CE2','CE3','CZ2','CZ3','CH2','NE1'],
    'HIS': ['CG','ND1','CD2','CE1','NE2'],
}

# Cation-PI residues
CATION_RES = {'ARG','LYS','HIS'}
CATION_PI_DIST = 6.0  # Å

# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def log(msg, verbose=True):
    """Timestamped logging."""
    if verbose:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def rad_to_deg(r):
    return math.degrees(r) if r is not None else None


def safe_div(a, b, default=0.0):
    return a / b if b != 0 else default


def get_atom_element(atom):
    """Robustly return element symbol from BioPython atom."""
    elem = atom.element
    if elem and elem.strip() and elem.strip() not in ('.', '?', ''):
        return elem.strip().upper()
    # Fall back to name-based guess
    name = atom.get_name().strip().lstrip('0123456789')
    return name[0].upper() if name else 'C'


def vector_angle(v1, v2):
    """Angle (degrees) between two numpy 3-vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def ring_centroid(residue, ring_def):
    """Return centroid of an aromatic ring."""
    pts = []
    for name in ring_def:
        if name in residue:
            pts.append(residue[name].get_vector().get_array())
    return np.mean(pts, axis=0) if len(pts) >= 3 else None


def ring_normal(residue, ring_def):
    """Return unit normal vector to an aromatic ring plane."""
    pts = []
    for name in ring_def:
        if name in residue:
            pts.append(residue[name].get_vector().get_array())
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    c = pts.mean(axis=0)
    # PCA for normal
    _, _, vh = np.linalg.svd(pts - c)
    return vh[-1]  # last singular vector = normal to plane


# ─── PARSER & CLEANER ────────────────────────────────────────────────────────

class PDBLoader:
    """
    Robustly loads any PDB file — full RCSB format (with HEADER, REMARK,
    SEQRES, HELIX, SHEET, SSBOND …) or stripped ATOM-only format.
    Detects format automatically and harmonises both into a uniform
    internal representation.
    """

    def __init__(self, pdb_path: str, verbose: bool = True):
        self.path    = Path(pdb_path)
        self.verbose = verbose
        self.parser  = PDBParser(QUIET=True)
        self.raw_text = self.path.read_text(errors='replace')
        self.pdb_type = self._detect_type()
        self.structure = self._load()
        self.model     = self.structure[0]

    def _detect_type(self):
        has_header = any(
            self.raw_text.startswith(rec)
            for rec in ('HEADER', 'REMARK', 'TITLE', 'ATOM  ', 'HETATM')
        )
        has_remark = 'REMARK' in self.raw_text
        has_seqres = 'SEQRES' in self.raw_text
        has_helix  = 'HELIX ' in self.raw_text or 'SHEET ' in self.raw_text
        if has_remark or has_seqres or has_helix:
            return 'full'
        return 'atoms_only'

    def _load(self):
        log(f"Loading PDB ({self.pdb_type} format): {self.path.name}",
            self.verbose)
        return self.parser.get_structure(self.path.stem, str(self.path))

    # ── raw record parsers ─────────────────────────────────────────────────

    def parse_raw_ssbonds(self):
        """Parse SSBOND records from raw text."""
        bonds = []
        for line in self.raw_text.splitlines():
            if line.startswith('SSBOND'):
                try:
                    res1_chain = line[15].strip()
                    res1_num   = int(line[17:21].strip())
                    res2_chain = line[29].strip()
                    res2_num   = int(line[31:35].strip())
                    dist = float(line[73:78].strip()) if len(line) > 73 else None
                    bonds.append({
                        'chain1': res1_chain, 'resnum1': res1_num,
                        'chain2': res2_chain, 'resnum2': res2_num,
                        'declared_dist': dist
                    })
                except (ValueError, IndexError):
                    pass
        return bonds

    def parse_raw_helices(self):
        """Parse HELIX records from raw text."""
        helices = []
        for line in self.raw_text.splitlines():
            if line.startswith('HELIX '):
                try:
                    hid   = line[7:10].strip()
                    chain = line[19].strip()
                    start = int(line[21:25].strip())
                    end   = int(line[33:37].strip())
                    htype = int(line[38:40].strip()) if len(line) > 39 else 1
                    length= int(line[71:76].strip()) if len(line) > 75 else end-start+1
                    helices.append({
                        'helix_id': hid, 'chain': chain,
                        'start': start, 'end': end,
                        'helix_type': htype, 'length': length
                    })
                except (ValueError, IndexError):
                    pass
        return helices

    def parse_raw_sheets(self):
        """Parse SHEET records from raw text."""
        sheets = []
        for line in self.raw_text.splitlines():
            if line.startswith('SHEET '):
                try:
                    strand  = int(line[7:10].strip())
                    sheet   = line[11:14].strip()
                    chain   = line[21].strip()
                    start   = int(line[22:26].strip())
                    end     = int(line[33:37].strip())
                    sense   = int(line[38:40].strip()) if len(line) > 39 else 0
                    sheets.append({
                        'strand': strand, 'sheet_id': sheet, 'chain': chain,
                        'start': start, 'end': end, 'sense': sense
                    })
                except (ValueError, IndexError):
                    pass
        return sheets

    def parse_raw_sites(self):
        """Parse SITE records (active/binding sites)."""
        sites = defaultdict(list)
        for line in self.raw_text.splitlines():
            if line.startswith('SITE  '):
                try:
                    site_id = line[11:14].strip()
                    # 4 residues per SITE record
                    for i in range(4):
                        offset = 18 + i * 11
                        if offset + 9 > len(line):
                            break
                        res = line[offset:offset+3].strip()
                        ch  = line[offset+4] if offset+4 < len(line) else ''
                        num_s = line[offset+5:offset+9].strip()
                        if res and num_s.lstrip('-').isdigit():
                            sites[site_id].append(
                                {'residue': res, 'chain': ch,
                                 'resnum': int(num_s)}
                            )
                except (ValueError, IndexError):
                    pass
        return dict(sites)

    def parse_seqres(self):
        """Parse SEQRES records into chain → sequence."""
        seqres = defaultdict(list)
        for line in self.raw_text.splitlines():
            if line.startswith('SEQRES'):
                try:
                    chain = line[11].strip()
                    residues = line[19:].split()
                    seqres[chain].extend(residues)
                except IndexError:
                    pass
        return dict(seqres)

    def parse_conect(self):
        """Parse CONECT records for explicit bonds."""
        bonds = []
        for line in self.raw_text.splitlines():
            if line.startswith('CONECT'):
                parts = line[6:].split()
                if len(parts) >= 2:
                    try:
                        a1 = int(parts[0])
                        for p in parts[1:]:
                            bonds.append((a1, int(p)))
                    except ValueError:
                        pass
        return bonds

    def parse_cryst1(self):
        """Parse CRYST1 unit cell parameters."""
        for line in self.raw_text.splitlines():
            if line.startswith('CRYST1'):
                try:
                    return {
                        'a': float(line[6:15]),  'b': float(line[15:24]),
                        'c': float(line[24:33]), 'alpha': float(line[33:40]),
                        'beta':  float(line[40:47]), 'gamma': float(line[47:54]),
                        'space_group': line[55:66].strip()
                    }
                except (ValueError, IndexError):
                    pass
        return {}

    def get_header_info(self):
        """Extract PDB header metadata."""
        h = self.structure.header
        cryst = self.parse_cryst1()
        return {
            'pdb_id':           self.structure.id,
            'pdb_type':         self.pdb_type,
            'title':            h.get('name', ''),
            'classification':   h.get('head', ''),
            'deposition_date':  h.get('deposition_date', ''),
            'resolution_A':     h.get('resolution', None),
            'structure_method': h.get('structure_method', ''),
            'author':           h.get('author', ''),
            'keywords':         h.get('keywords', ''),
            'journal':          str(h.get('journal_reference', ''))[:200],
            'unit_cell_a':      cryst.get('a'),
            'unit_cell_b':      cryst.get('b'),
            'unit_cell_c':      cryst.get('c'),
            'unit_cell_alpha':  cryst.get('alpha'),
            'unit_cell_beta':   cryst.get('beta'),
            'unit_cell_gamma':  cryst.get('gamma'),
            'space_group':      cryst.get('space_group', ''),
        }

# ─── FEATURE EXTRACTION MODULES ──────────────────────────────────────────────

class GlobalFeatureExtractor:
    """
    Extract protein-wide / global-level structural features.
    Covers composition, shape, geometry, thermodynamic estimates.
    """

    def __init__(self, loader: PDBLoader):
        self.loader  = loader
        self.model   = loader.model
        self.aa_res  = [r for chain in self.model
                        for r in chain.get_residues()
                        if is_aa(r, standard=False)]
        self.all_atoms = list(self.model.get_atoms())
        self.all_heavy = [a for a in self.all_atoms
                          if get_atom_element(a) != 'H']

    # ── composition ──────────────────────────────────────────────────────────

    def sequence_composition(self):
        """Amino acid counts, frequencies, physicochemical ratios."""
        counts = Counter(r.get_resname() for r in self.aa_res)
        total  = sum(counts.values())
        feats  = {'n_residues': total}

        for aa in ONE_LETTER:
            c = counts.get(aa, 0)
            feats[f'count_{aa}'] = c
            feats[f'frac_{aa}'] = safe_div(c, total)

        # Class fractions
        class_counts = Counter()
        for r in self.aa_res:
            cls = RESIDUE_CLASS.get(r.get_resname(), 'other')
            class_counts[cls] += 1

        for cls in set(RESIDUE_CLASS.values()):
            feats[f'frac_{cls}'] = safe_div(class_counts.get(cls, 0), total)

        # Molecular weight
        feats['mw_da'] = sum(
            RESIDUE_MW.get(r.get_resname(), 110.0) for r in self.aa_res
        )
        # Net charge at pH 7
        feats['net_charge_pH7'] = sum(
            FORMAL_CHARGE.get(r.get_resname(), 0) for r in self.aa_res
        )
        feats['n_positive'] = sum(
            1 for r in self.aa_res
            if FORMAL_CHARGE.get(r.get_resname(), 0) > 0
        )
        feats['n_negative'] = sum(
            1 for r in self.aa_res
            if FORMAL_CHARGE.get(r.get_resname(), 0) < 0
        )
        feats['charge_ratio'] = safe_div(
            feats['n_positive'],
            feats['n_positive'] + feats['n_negative']
        )
        # Hydrophobic fraction (KD > 0)
        feats['frac_hydrophobic'] = safe_div(
            sum(1 for r in self.aa_res
                if KD_HYDROPHOBICITY.get(r.get_resname(), 0) > 0),
            total
        )
        # Mean physicochemical properties
        feats['mean_hydrophobicity_KD'] = np.mean([
            KD_HYDROPHOBICITY.get(r.get_resname(), 0)
            for r in self.aa_res
        ]) if total > 0 else 0
        feats['mean_hydrophobicity_Eisenberg'] = np.mean([
            EISENBERG_HYDROPHOBICITY.get(r.get_resname(), 0)
            for r in self.aa_res
        ]) if total > 0 else 0
        feats['mean_residue_volume'] = np.mean([
            VDW_VOLUME.get(r.get_resname(), 100)
            for r in self.aa_res
        ]) if total > 0 else 0

        # Isoelectric point approximation (Henderson-Hasselbalch)
        feats['pI_approx'] = self._estimate_pI()
        return feats

    def _estimate_pI(self):
        """Approximate pI via charge vs pH titration."""
        pka_vals = []
        for r in self.aa_res:
            name = r.get_resname()
            if name in SIDE_CHAIN_PKA:
                pka_vals.append((name, SIDE_CHAIN_PKA[name]))

        def net_charge_at_pH(pH):
            q = 0.0
            q += 1.0 / (1 + 10**(pH - 8.0))   # N-terminus
            q -= 1.0 / (1 + 10**(3.1 - pH))    # C-terminus
            for name, pka in pka_vals:
                if name in ('ASP', 'GLU'):
                    q -= 1.0 / (1 + 10**(pka - pH))
                elif name in ('LYS', 'ARG', 'HIS'):
                    q += 1.0 / (1 + 10**(pH - pka))
                elif name == 'CYS':
                    q -= 1.0 / (1 + 10**(pka - pH))
                elif name == 'TYR':
                    q -= 1.0 / (1 + 10**(pka - pH))
            return q

        lo, hi = 0.0, 14.0
        for _ in range(150):
            mid = (lo + hi) / 2
            if net_charge_at_pH(mid) > 0:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2, 3)

    # ── geometry / shape ─────────────────────────────────────────────────────

    def geometric_features(self):
        """Centre of mass, radius of gyration, convex hull, PCA axes."""
        ca_coords = np.array([
            r['CA'].get_vector().get_array()
            for r in self.aa_res if 'CA' in r
        ])
        all_coords = np.array([a.get_vector().get_array()
                                for a in self.all_heavy])

        feats = {}
        # Centre of mass (heavy atoms)
        masses = np.array([
            12.0 if get_atom_element(a) == 'C' else
            14.0 if get_atom_element(a) == 'N' else
            16.0 if get_atom_element(a) == 'O' else
            32.0 if get_atom_element(a) == 'S' else 12.0
            for a in self.all_heavy
        ])
        com = np.average(all_coords, axis=0, weights=masses)
        feats['com_x'], feats['com_y'], feats['com_z'] = com

        # Radius of gyration (CA)
        centroid_ca = ca_coords.mean(axis=0)
        rg = np.sqrt(np.mean(np.sum((ca_coords - centroid_ca)**2, axis=1)))
        feats['radius_of_gyration_A'] = rg

        # End-to-end distance (N-to-C terminus CA)
        if len(ca_coords) >= 2:
            feats['end_to_end_dist_A'] = np.linalg.norm(
                ca_coords[-1] - ca_coords[0]
            )
        else:
            feats['end_to_end_dist_A'] = 0.0

        # Contour length (sum of consecutive CA distances)
        feats['contour_length_A'] = float(np.sum(
            np.linalg.norm(np.diff(ca_coords, axis=0), axis=1)
        ))

        # Convex hull on CA coords
        try:
            hull = ConvexHull(ca_coords)
            feats['convex_hull_volume_A3'] = hull.volume
            feats['convex_hull_area_A2']   = hull.area
            feats['packing_ratio'] = safe_div(rg**3 * (4/3) * math.pi,
                                              hull.volume)
        except Exception:
            feats['convex_hull_volume_A3'] = 0
            feats['convex_hull_area_A2']   = 0
            feats['packing_ratio']         = 0

        # PCA on CA coordinates
        cov = np.cov((ca_coords - centroid_ca).T)
        eigvals = sorted(np.linalg.eigh(cov)[0], reverse=True)
        feats['pca_eigenval_1'] = eigvals[0]
        feats['pca_eigenval_2'] = eigvals[1]
        feats['pca_eigenval_3'] = eigvals[2]

        lam = eigvals
        s = lam[0] + lam[1] + lam[2]
        feats['asphericity'] = lam[0] - 0.5 * (lam[1] + lam[2])
        feats['acylindricity'] = lam[1] - lam[2]
        feats['relative_shape_anisotropy'] = safe_div(
            feats['asphericity']**2 + 0.75 * feats['acylindricity']**2,
            s**2
        ) if s > 0 else 0

        # Prolateness
        feats['prolateness'] = safe_div(
            lam[0] - 0.5 * (lam[1] + lam[2]),
            s
        )

        # Max pairwise CA distance (protein diameter)
        dists = cdist(ca_coords, ca_coords)
        feats['max_pairwise_dist_A'] = float(dists.max())
        feats['mean_pairwise_dist_A'] = float(
            dists[np.triu_indices_from(dists, k=1)].mean()
        )

        # Number of atoms
        feats['n_heavy_atoms'] = len(self.all_heavy)
        feats['n_total_atoms'] = len(self.all_atoms)
        feats['n_chains'] = len(list(self.model.get_chains()))
        feats['n_water'] = sum(
            1 for r in self.model.get_residues()
            if r.get_resname() in ('HOH', 'WAT', 'H2O')
        )
        feats['n_hetatm'] = sum(
            1 for r in self.model.get_residues()
            if not is_aa(r, standard=False)
            and r.get_resname() not in ('HOH','WAT','H2O')
        )

        return feats

    # ── b-factors / thermal motion ────────────────────────────────────────────

    def bfactor_features(self):
        """Global B-factor statistics."""
        bfs_all = [a.get_bfactor() for a in self.all_heavy]
        bfs_ca  = [r['CA'].get_bfactor() for r in self.aa_res if 'CA' in r]
        feats   = {}
        for tag, vals in [('all_heavy', bfs_all), ('CA', bfs_ca)]:
            if vals:
                feats[f'bfactor_{tag}_mean']   = np.mean(vals)
                feats[f'bfactor_{tag}_std']    = np.std(vals)
                feats[f'bfactor_{tag}_min']    = np.min(vals)
                feats[f'bfactor_{tag}_max']    = np.max(vals)
                feats[f'bfactor_{tag}_median'] = np.median(vals)
                feats[f'bfactor_{tag}_skew']   = float(skew(vals))
                feats[f'bfactor_{tag}_kurt']   = float(kurtosis(vals))
                feats[f'bfactor_{tag}_range']  = np.max(vals) - np.min(vals)
                feats[f'bfactor_{tag}_IQR'] = (
                    float(np.percentile(vals, 75)) -
                    float(np.percentile(vals, 25))
                )
        return feats

    # ── secondary structure summary ───────────────────────────────────────────

    def secondary_structure_summary(self, ss_assignments: dict):
        """Aggregate secondary structure across the whole structure."""
        total = len(self.aa_res)
        counts = Counter(ss_assignments.values())
        feats = {
            'ss_helix_count':   counts.get('H', 0),
            'ss_sheet_count':   counts.get('E', 0),
            'ss_coil_count':    counts.get('-', 0),
            'ss_frac_helix':    safe_div(counts.get('H', 0), total),
            'ss_frac_sheet':    safe_div(counts.get('E', 0), total),
            'ss_frac_coil':     safe_div(counts.get('-', 0), total),
        }
        # Run-length segments
        values = [ss_assignments.get(k, '-') for k in sorted(
            ss_assignments.keys(), key=lambda x: x[1]
        )]
        # Count contiguous helix / sheet segments
        def count_runs(lst, val):
            runs = 0
            in_run = False
            for v in lst:
                if v == val and not in_run:
                    runs += 1
                    in_run = True
                elif v != val:
                    in_run = False
            return runs

        feats['ss_n_helix_segments'] = count_runs(values, 'H')
        feats['ss_n_sheet_segments'] = count_runs(values, 'E')
        feats['ss_n_coil_segments']  = count_runs(values, '-')
        return feats

    def run(self, ss_assignments):
        feats = {}
        feats.update(self.loader.get_header_info())
        feats.update(self.sequence_composition())
        feats.update(self.geometric_features())
        feats.update(self.bfactor_features())
        feats.update(self.secondary_structure_summary(ss_assignments))
        return feats


# ─────────────────────────────────────────────────────────────────────────────

class PerResidueExtractor:
    """
    Exhaustive per-residue feature extraction.
    Produces one row per residue covering ~120 features.
    """

    def __init__(self, loader: PDBLoader):
        self.loader  = loader
        self.model   = loader.model
        self.aa_res  = []
        self.chain_res_map = defaultdict(list)
        for chain in self.model:
            for r in chain.get_residues():
                if is_aa(r, standard=False):
                    self.aa_res.append(r)
                    self.chain_res_map[chain.id].append(r)

        # Build atom index for NeighborSearch
        all_atoms = [a for a in self.model.get_atoms()
                     if get_atom_element(a) != 'H']
        self.ns = NeighborSearch(all_atoms)

    # ── SASA via freesasa ─────────────────────────────────────────────────────

    def compute_sasa(self):
        """Per-residue SASA (absolute and relative) using FreeSASA."""
        sasa_map = {}  # (chain, resnum) -> {'total', 'backbone', 'sidechain', 'relative'}
        try:
            fs_struct = freesasa.Structure(str(self.loader.path))
            fs_result = freesasa.calc(fs_struct)
            n = fs_struct.nAtoms()

            atom_sasa = {i: fs_result.atomArea(i) for i in range(n)}
            # Map atom serial → sasa
            # freesasa atoms are in PDB order; we map by residue
            res_sasa = defaultdict(lambda: {'total': 0.0, 'backbone': 0.0, 'sidechain': 0.0})
            backbone_names = {'N','CA','C','O','OXT'}

            for i in range(n):
                chain   = fs_struct.chainLabel(i)
                resnum  = int(fs_struct.residueNumber(i).strip())
                aname   = fs_struct.atomName(i).strip()
                area    = atom_sasa[i]
                key     = (chain, resnum)
                res_sasa[key]['total'] += area
                if aname in backbone_names:
                    res_sasa[key]['backbone'] += area
                else:
                    res_sasa[key]['sidechain'] += area

            for r in self.aa_res:
                ch  = r.get_parent().id
                num = r.get_id()[1]
                key = (ch, num)
                d   = res_sasa.get(key, {'total':0,'backbone':0,'sidechain':0})
                max_s = MAX_SASA.get(r.get_resname(), 200)
                sasa_map[(ch, num)] = {
                    'sasa_total':     d['total'],
                    'sasa_backbone':  d['backbone'],
                    'sasa_sidechain': d['sidechain'],
                    'sasa_relative':  safe_div(d['total'], max_s),
                    'is_buried':      int(safe_div(d['total'], max_s) < 0.25),
                    'is_surface':     int(safe_div(d['total'], max_s) > 0.40),
                }
        except Exception as e:
            for r in self.aa_res:
                ch  = r.get_parent().id
                num = r.get_id()[1]
                sasa_map[(ch, num)] = {
                    'sasa_total': None, 'sasa_backbone': None,
                    'sasa_sidechain': None, 'sasa_relative': None,
                    'is_buried': None, 'is_surface': None,
                }
        return sasa_map

    # ── DSSP secondary structure ──────────────────────────────────────────────

    def compute_dssp(self):
        """
        Assign secondary structure using pydssp.
        Returns dict: (chain, resnum) → SS code ('H','E','-')
        """
        ss_map = {}
        try:
            coords = dssp_read(self.loader.raw_text)
            ss_arr = dssp_assign(coords)
            for i, r in enumerate(self.aa_res):
                ch  = r.get_parent().id
                num = r.get_id()[1]
                ss_map[(ch, num)] = ss_arr[i] if i < len(ss_arr) else '-'
        except Exception:
            for r in self.aa_res:
                ss_map[(r.get_parent().id, r.get_id()[1])] = '-'
        return ss_map

    # ── phi/psi/omega/chi dihedrals ───────────────────────────────────────────

    def compute_dihedrals(self):
        """
        Phi, Psi, Omega backbone + Chi1/Chi2 side-chain dihedrals.
        """
        dihedral_map = {}
        ppb = PPBuilder()

        for pp in ppb.build_peptides(self.model):
            phipsi = pp.get_phi_psi_list()
            for i, r in enumerate(pp):
                ch  = r.get_parent().id
                num = r.get_id()[1]
                phi, psi = phipsi[i]
                d = {
                    'phi_deg': rad_to_deg(phi),
                    'psi_deg': rad_to_deg(psi),
                    'omega_deg': None,
                    'chi1_deg': None,
                    'chi2_deg': None,
                }

                # Omega angle (CA-C-N-CA across peptide bond)
                if i > 0:
                    try:
                        prev = pp[i-1]
                        v1 = prev['CA'].get_vector()
                        v2 = prev['C'].get_vector()
                        v3 = r['N'].get_vector()
                        v4 = r['CA'].get_vector()
                        d['omega_deg'] = rad_to_deg(
                            calc_dihedral(v1, v2, v3, v4)
                        )
                    except (KeyError, Exception):
                        pass

                # Chi1
                chi1_def = CHI1_ATOMS.get(r.get_resname())
                if chi1_def:
                    try:
                        vecs = [r[a].get_vector() for a in chi1_def]
                        d['chi1_deg'] = rad_to_deg(
                            calc_dihedral(*vecs)
                        )
                    except (KeyError, Exception):
                        pass

                # Chi2
                chi2_def = CHI2_ATOMS.get(r.get_resname())
                if chi2_def:
                    try:
                        vecs = [r[a].get_vector() for a in chi2_def]
                        d['chi2_deg'] = rad_to_deg(
                            calc_dihedral(*vecs)
                        )
                    except (KeyError, Exception):
                        pass

                dihedral_map[(ch, num)] = d

        return dihedral_map

    # ── Ramachandran region ───────────────────────────────────────────────────

    @staticmethod
    def ramachandran_region(phi, psi):
        """Classify (phi, psi) into Ramachandran regions."""
        if phi is None or psi is None:
            return 'undefined'
        # Core alpha-helix
        if -160 < phi < -20 and -80 < psi < 50:
            return 'alpha_helix'
        # Beta-sheet
        if (-180 < phi < -50 and (100 < psi <= 180 or -180 <= psi < -100)):
            return 'beta_sheet'
        if -180 < phi < -50 and 100 < psi <= 180:
            return 'beta_sheet'
        # Left-handed alpha
        if 20 < phi < 120 and 20 < psi < 120:
            return 'left_alpha'
        # Polyproline II
        if -90 < phi < -50 and 120 < psi <= 180:
            return 'ppII'
        return 'allowed_other'

    # ── contact environment ───────────────────────────────────────────────────

    def compute_contacts(self):
        """
        Per-residue contact features:
          - n_contacts_CA8  : CA-CA contacts < 8Å
          - n_contacts_heavy5 : heavy-heavy contacts < 5Å
          - n_local_contacts : |i-j| <= 4
          - n_medium_contacts: 5 <= |i-j| <= 11
          - n_long_contacts : |i-j| >= 12
        """
        res_index = {(r.get_parent().id, r.get_id()[1]): i
                     for i, r in enumerate(self.aa_res)}

        contact_feats = defaultdict(lambda: {
            'n_contacts_CA8': 0, 'n_contacts_heavy5': 0,
            'n_local_contacts': 0, 'n_medium_contacts': 0,
            'n_long_contacts': 0, 'n_hydrophobic_contacts': 0,
            'mean_contact_dist': 0.0, 'contact_order_local': 0,
        })

        ca_coords = {}
        for r in self.aa_res:
            if 'CA' in r:
                ca_coords[(r.get_parent().id, r.get_id()[1])] = \
                    r['CA'].get_vector().get_array()

        keys = list(ca_coords.keys())
        coords_arr = np.array([ca_coords[k] for k in keys])
        dmat_full = cdist(coords_arr, coords_arr)

        for i, ki in enumerate(keys):
            ri = res_index[ki]
            dists = []
            for j, kj in enumerate(keys):
                if i == j:
                    continue
                d = dmat_full[i, j]
                sep = abs(ri - res_index.get(kj, ri))
                if d < CA_CONTACT_DIST:
                    contact_feats[ki]['n_contacts_CA8'] += 1
                    dists.append(d)
                    if sep <= 4:
                        contact_feats[ki]['n_local_contacts'] += 1
                    elif sep <= 11:
                        contact_feats[ki]['n_medium_contacts'] += 1
                    else:
                        contact_feats[ki]['n_long_contacts'] += 1

                    # Hydrophobic contacts
                    rj_res = self.aa_res[res_index.get(kj, 0)]
                    if (self.aa_res[ri].get_resname() in HYDROPHOBIC_RES and
                            rj_res.get_resname() in HYDROPHOBIC_RES):
                        contact_feats[ki]['n_hydrophobic_contacts'] += 1

            if dists:
                contact_feats[ki]['mean_contact_dist'] = np.mean(dists)

        # Heavy-atom contacts (< 5Å) using NeighborSearch
        for r in self.aa_res:
            key = (r.get_parent().id, r.get_id()[1])
            heavy = [a for a in r.get_atoms() if get_atom_element(a) != 'H']
            n_heavy_contacts = 0
            for a in heavy:
                pos_arr = a.get_vector().get_array()
                neighbours = self.ns.search(pos_arr, HEAVY_CONTACT_DIST,
                                             level='A')
                for nb in neighbours:
                    nb_res = nb.get_parent()
                    nb_key = (nb_res.get_parent().id, nb_res.get_id()[1])
                    if nb_key != key:
                        n_heavy_contacts += 1
            contact_feats[key]['n_contacts_heavy5'] = n_heavy_contacts

        return dict(contact_feats)

    # ── hydrogen bonds (geometry-based) ──────────────────────────────────────

    def compute_hbonds(self):
        """
        Detect hydrogen bonds using donor/acceptor geometry.
        N-O distance < 3.5Å, rough D-A-N angle filter.
        Returns per-residue donor and acceptor counts.
        """
        donors    = {'N', 'NE', 'NH1', 'NH2', 'NZ', 'ND1', 'NE2', 'ND2',
                     'NE1', 'OG', 'OG1', 'OH', 'OE2', 'OD1'}
        acceptors = {'O', 'OD1', 'OD2', 'OE1', 'OE2', 'ND1', 'NE2',
                     'OG', 'OG1', 'OH', 'SD'}

        donor_atoms = []
        accept_atoms = []
        for r in self.aa_res:
            for a in r.get_atoms():
                if a.get_name().strip() in donors:
                    donor_atoms.append(a)
                if a.get_name().strip() in acceptors:
                    accept_atoms.append(a)

        hb_donor = Counter()
        hb_acceptor = Counter()

        for d_atom in donor_atoms:
            d_res = d_atom.get_parent()
            d_key = (d_res.get_parent().id, d_res.get_id()[1])
            for a_atom in accept_atoms:
                a_res = a_atom.get_parent()
                a_key = (a_res.get_parent().id, a_res.get_id()[1])
                if d_key == a_key:
                    continue
                dist = (d_atom.get_vector() - a_atom.get_vector()).norm()
                if dist <= HBOND_DA_DIST:
                    hb_donor[d_key] += 1
                    hb_acceptor[a_key] += 1

        result = {}
        for r in self.aa_res:
            key = (r.get_parent().id, r.get_id()[1])
            result[key] = {
                'n_hbond_donor': hb_donor.get(key, 0),
                'n_hbond_acceptor': hb_acceptor.get(key, 0),
                'n_hbond_total': hb_donor.get(key, 0) + hb_acceptor.get(key, 0),
            }
        return result

    # ── salt bridges ──────────────────────────────────────────────────────────

    def compute_salt_bridges(self):
        """
        Detect salt bridges between oppositely charged residues (< 4Å).
        """
        pos_atoms = {
            'ARG': ['NH1','NH2','NE'],
            'LYS': ['NZ'],
            'HIS': ['ND1','NE2'],
        }
        neg_atoms = {
            'ASP': ['OD1','OD2'],
            'GLU': ['OE1','OE2'],
        }

        pos_list, neg_list = [], []
        for r in self.aa_res:
            rn = r.get_resname()
            for aname in pos_atoms.get(rn, []):
                if aname in r:
                    pos_list.append((r, r[aname]))
            for aname in neg_atoms.get(rn, []):
                if aname in r:
                    neg_list.append((r, r[aname]))

        salt_count = Counter()
        for pr, pa in pos_list:
            for nr, na in neg_list:
                d = (pa.get_vector() - na.get_vector()).norm()
                if d <= SALTBRIDGE_DIST:
                    salt_count[(pr.get_parent().id, pr.get_id()[1])] += 1
                    salt_count[(nr.get_parent().id, nr.get_id()[1])] += 1

        result = {}
        for r in self.aa_res:
            key = (r.get_parent().id, r.get_id()[1])
            result[key] = {'n_salt_bridges': salt_count.get(key, 0)}
        return result

    # ── disulfide bonds ───────────────────────────────────────────────────────

    def compute_disulfides(self):
        """
        Detect CYS-CYS disulfide bonds by SG-SG distance < 2.2 Å
        and flag each CYS residue.
        """
        cys_res = [r for r in self.aa_res if r.get_resname() == 'CYS'
                   and 'SG' in r]
        in_disulfide = set()
        ss_partners  = {}

        for i, ri in enumerate(cys_res):
            for j, rj in enumerate(cys_res):
                if j <= i:
                    continue
                d = (ri['SG'].get_vector() - rj['SG'].get_vector()).norm()
                if d <= SSBOND_DIST:
                    ki = (ri.get_parent().id, ri.get_id()[1])
                    kj = (rj.get_parent().id, rj.get_id()[1])
                    in_disulfide.add(ki)
                    in_disulfide.add(kj)
                    ss_partners[ki] = kj
                    ss_partners[kj] = ki

        result = {}
        for r in self.aa_res:
            key = (r.get_parent().id, r.get_id()[1])
            partner = ss_partners.get(key)
            result[key] = {
                'in_disulfide': int(key in in_disulfide),
                'ss_partner_chain': partner[0] if partner else '',
                'ss_partner_resnum': partner[1] if partner else -1,
            }
        return result

    # ── pi-pi stacking ────────────────────────────────────────────────────────

    def compute_pipi(self):
        """
        Detect pi-pi stacking between aromatic residues.
        Criteria: centroid dist < 7Å, dihedral between ring normals < 30°.
        """
        aro_res = [r for r in self.aa_res
                   if r.get_resname() in RING_ATOMS]
        pipi_count = Counter()

        for i, ri in enumerate(aro_res):
            rdef_i  = RING_ATOMS[ri.get_resname()]
            cent_i  = ring_centroid(ri, rdef_i)
            norm_i  = ring_normal(ri, rdef_i)
            if cent_i is None or norm_i is None:
                continue
            for j, rj in enumerate(aro_res):
                if j <= i:
                    continue
                rdef_j  = RING_ATOMS[rj.get_resname()]
                cent_j  = ring_centroid(rj, rdef_j)
                norm_j  = ring_normal(rj, rdef_j)
                if cent_j is None or norm_j is None:
                    continue
                d = np.linalg.norm(cent_i - cent_j)
                if d < PIPI_DIST:
                    ang = vector_angle(norm_i, norm_j)
                    if ang is not None:
                        ang = min(ang, 180 - ang)
                        if ang < PIPI_ANGLE:  # parallel
                            ki = (ri.get_parent().id, ri.get_id()[1])
                            kj = (rj.get_parent().id, rj.get_id()[1])
                            pipi_count[ki] += 1
                            pipi_count[kj] += 1

        result = {}
        for r in self.aa_res:
            key = (r.get_parent().id, r.get_id()[1])
            result[key] = {'n_pipi_stacking': pipi_count.get(key, 0)}
        return result

    # ── cation-pi interactions ────────────────────────────────────────────────

    def compute_cation_pi(self):
        """Cation-PI interactions (within 6Å of ring centroid)."""
        aro_res  = [r for r in self.aa_res if r.get_resname() in RING_ATOMS]
        cat_res  = [r for r in self.aa_res if r.get_resname() in CATION_RES]
        count    = Counter()

        for raro in aro_res:
            rdef   = RING_ATOMS[raro.get_resname()]
            cent   = ring_centroid(raro, rdef)
            if cent is None:
                continue
            for rcat in cat_res:
                if rcat == raro:
                    continue
                for a in rcat.get_atoms():
                    d = np.linalg.norm(a.get_vector().get_array() - cent)
                    if d < CATION_PI_DIST:
                        ka = (raro.get_parent().id, raro.get_id()[1])
                        kc = (rcat.get_parent().id, rcat.get_id()[1])
                        count[ka] += 1
                        count[kc] += 1
                        break

        result = {}
        for r in self.aa_res:
            key = (r.get_parent().id, r.get_id()[1])
            result[key] = {'n_cation_pi': count.get(key, 0)}
        return result

    # ── backbone geometry ─────────────────────────────────────────────────────

    def compute_backbone_geometry(self):
        """
        Per-residue backbone bond lengths and angles.
        CA-N, CA-C, C-N(next), N-CA-C, CA-C-N angles.
        """
        result = {}
        for i, r in enumerate(self.aa_res):
            key = (r.get_parent().id, r.get_id()[1])
            d = {
                'bond_CA_N':   None, 'bond_CA_C':   None, 'bond_C_O':    None,
                'bond_N_CA':   None, 'angle_N_CA_C': None,
                'angle_CA_C_O': None, 'ca_displacement': None,
            }
            try:
                if 'CA' in r and 'N' in r:
                    d['bond_CA_N'] = (r['CA'].get_vector() - r['N'].get_vector()).norm()
                if 'CA' in r and 'C' in r:
                    d['bond_CA_C'] = (r['CA'].get_vector() - r['C'].get_vector()).norm()
                if 'C' in r and 'O' in r:
                    d['bond_C_O'] = (r['C'].get_vector() - r['O'].get_vector()).norm()
                if 'N' in r and 'CA' in r and 'C' in r:
                    d['angle_N_CA_C'] = rad_to_deg(
                        calc_angle(r['N'].get_vector(),
                                   r['CA'].get_vector(),
                                   r['C'].get_vector())
                    )
                if 'CA' in r and 'C' in r and 'O' in r:
                    d['angle_CA_C_O'] = rad_to_deg(
                        calc_angle(r['CA'].get_vector(),
                                   r['C'].get_vector(),
                                   r['O'].get_vector())
                    )
                # CA displacement from ideal (adjacent CA midpoint)
                if (i > 0 and i < len(self.aa_res)-1 and
                        'CA' in r and
                        'CA' in self.aa_res[i-1] and
                        'CA' in self.aa_res[i+1]):
                    prev_ca = self.aa_res[i-1]['CA'].get_vector().get_array()
                    next_ca = self.aa_res[i+1]['CA'].get_vector().get_array()
                    mid  = (prev_ca + next_ca) / 2
                    curr = r['CA'].get_vector().get_array()
                    d['ca_displacement'] = float(np.linalg.norm(curr - mid))
            except Exception:
                pass
            result[key] = d
        return result

    # ── B-factor per residue ──────────────────────────────────────────────────

    def compute_residue_bfactors(self):
        """Per-residue B-factor statistics (mean, max of all atoms)."""
        result = {}
        for r in self.aa_res:
            key = (r.get_parent().id, r.get_id()[1])
            bfs = [a.get_bfactor() for a in r.get_atoms()]
            if bfs:
                result[key] = {
                    'bfactor_res_mean': np.mean(bfs),
                    'bfactor_res_max':  np.max(bfs),
                    'bfactor_res_std':  np.std(bfs),
                    'bfactor_backbone': np.mean(
                        [a.get_bfactor() for a in r.get_atoms()
                         if a.get_name().strip() in ('N','CA','C','O')]
                    ) if any(a.get_name().strip() in ('N','CA','C','O')
                              for a in r.get_atoms()) else None,
                    'bfactor_sidechain': np.mean(
                        [a.get_bfactor() for a in r.get_atoms()
                         if a.get_name().strip() not in ('N','CA','C','O')]
                    ) if any(a.get_name().strip() not in ('N','CA','C','O')
                              for a in r.get_atoms()) else None,
                }
            else:
                result[key] = {
                    'bfactor_res_mean': None, 'bfactor_res_max': None,
                    'bfactor_res_std': None, 'bfactor_backbone': None,
                    'bfactor_sidechain': None,
                }
        return result

    # ── network centrality ────────────────────────────────────────────────────

    def compute_network(self):
        """
        Build protein contact network (CB-CB < 8Å) and compute
        per-residue centrality metrics.
        """
        G = nx.Graph()
        for i, r in enumerate(self.aa_res):
            G.add_node(i)

        idx_map = {(r.get_parent().id, r.get_id()[1]): i
                   for i, r in enumerate(self.aa_res)}

        for i, ri in enumerate(self.aa_res):
            ref_i = 'CB' if 'CB' in ri else ('CA' if 'CA' in ri else None)
            if not ref_i:
                continue
            vi = ri[ref_i].get_vector().get_array()
            for j, rj in enumerate(self.aa_res):
                if j <= i:
                    continue
                ref_j = 'CB' if 'CB' in rj else ('CA' if 'CA' in rj else None)
                if not ref_j:
                    continue
                vj = rj[ref_j].get_vector().get_array()
                d = np.linalg.norm(vi - vj)
                if d < CB_CONTACT_DIST:
                    G.add_edge(i, j, weight=1.0/d)

        deg_cent  = nx.degree_centrality(G)
        bet_cent  = nx.betweenness_centrality(G, weight='weight', normalized=True)
        clo_cent  = nx.closeness_centrality(G)
        clust     = nx.clustering(G)
        try:
            eig_cent = nx.eigenvector_centrality(G, max_iter=1000,
                                                  weight='weight')
        except Exception:
            eig_cent = {n: 0.0 for n in G.nodes()}

        result = {}
        for r in self.aa_res:
            key = (r.get_parent().id, r.get_id()[1])
            idx = idx_map.get(key, 0)
            result[key] = {
                'network_degree_centrality':      deg_cent.get(idx, 0),
                'network_betweenness_centrality': bet_cent.get(idx, 0),
                'network_closeness_centrality':   clo_cent.get(idx, 0),
                'network_clustering_coeff':       clust.get(idx, 0),
                'network_eigenvector_centrality': eig_cent.get(idx, 0),
                'network_degree':                 G.degree(idx),
            }
        return result

    # ── water contacts ────────────────────────────────────────────────────────

    def compute_water_contacts(self):
        """Number of water molecules within 3.5Å of each residue."""
        water_atoms = [
            a for r in self.model.get_residues()
            if r.get_resname() in ('HOH','WAT','H2O')
            for a in r.get_atoms()
        ]
        result = {}
        for r in self.aa_res:
            key   = (r.get_parent().id, r.get_id()[1])
            count = 0
            min_d = 999.0
            for atom in r.get_atoms():
                for w in water_atoms:
                    d = (atom.get_vector() - w.get_vector()).norm()
                    if d < 3.5:
                        count += 1
                    if d < min_d:
                        min_d = d
            result[key] = {
                'n_water_contacts': count,
                'min_water_dist_A': min_d if min_d < 999 else None,
            }
        return result

    # ── neighbour composition ─────────────────────────────────────────────────

    def compute_neighbour_composition(self):
        """
        For each residue, fraction of each class in its 8Å CA sphere.
        """
        ca_coords = {}
        for r in self.aa_res:
            if 'CA' in r:
                ca_coords[(r.get_parent().id, r.get_id()[1])] = (
                    r, r['CA'].get_vector().get_array()
                )

        classes = list(set(RESIDUE_CLASS.values()))
        result  = {}

        for k, (ri, ci) in ca_coords.items():
            neighbours = []
            for k2, (rj, cj) in ca_coords.items():
                if k == k2:
                    continue
                if np.linalg.norm(ci - cj) < CA_CONTACT_DIST:
                    neighbours.append(rj.get_resname())

            total_n = len(neighbours)
            d = {'n_neighbours_8A': total_n}
            for cls in classes:
                cnt = sum(1 for nm in neighbours
                          if RESIDUE_CLASS.get(nm, '') == cls)
                d[f'neighbour_frac_{cls}'] = safe_div(cnt, total_n)
            result[k] = d

        return result

    # ── MASTER assembler ──────────────────────────────────────────────────────

    def run(self):
        """
        Run all per-residue extractors and merge into a list of row dicts.
        """
        sasa_map     = self.compute_sasa()
        ss_map       = self.compute_dssp()
        dihedral_map = self.compute_dihedrals()
        contact_map  = self.compute_contacts()
        hbond_map    = self.compute_hbonds()
        sb_map       = self.compute_salt_bridges()
        ss_bond_map  = self.compute_disulfides()
        pipi_map     = self.compute_pipi()
        catpi_map    = self.compute_cation_pi()
        bb_geom_map  = self.compute_backbone_geometry()
        bf_map       = self.compute_residue_bfactors()
        net_map      = self.compute_network()
        wat_map      = self.compute_water_contacts()
        nbcomp_map   = self.compute_neighbour_composition()

        rows = []
        for i, r in enumerate(self.aa_res):
            ch   = r.get_parent().id
            num  = r.get_id()[1]
            ins  = r.get_id()[2].strip()
            name = r.get_resname()
            key  = (ch, num)

            phi  = dihedral_map.get(key, {}).get('phi_deg')
            psi  = dihedral_map.get(key, {}).get('psi_deg')
            ss   = ss_map.get(key, '-')

            row = {
                # Identity
                'chain':       ch,
                'resnum':      num,
                'icode':       ins,
                'resname':     name,
                'resname_1L':  ONE_LETTER.get(name, 'X'),
                'residue_idx': i,

                # Physicochemistry
                'hydrophobicity_KD':        KD_HYDROPHOBICITY.get(name),
                'hydrophobicity_Eisenberg': EISENBERG_HYDROPHOBICITY.get(name),
                'residue_mw':               RESIDUE_MW.get(name),
                'formal_charge':            FORMAL_CHARGE.get(name),
                'vdw_volume':               VDW_VOLUME.get(name),
                'residue_class':            RESIDUE_CLASS.get(name, 'other'),
                'side_chain_pKa':           SIDE_CHAIN_PKA.get(name),

                # Secondary structure
                'ss_pydssp':               ss,
                'is_helix':                int(ss == 'H'),
                'is_sheet':                int(ss == 'E'),
                'is_coil':                 int(ss == '-'),
                'ramachandran_region':     self.ramachandran_region(phi, psi),

                # Dihedrals
                'phi_deg':                 phi,
                'psi_deg':                 psi,
                'omega_deg':               dihedral_map.get(key, {}).get('omega_deg'),
                'chi1_deg':                dihedral_map.get(key, {}).get('chi1_deg'),
                'chi2_deg':                dihedral_map.get(key, {}).get('chi2_deg'),

                # Occupancy
                'mean_occupancy':          np.mean(
                    [a.get_occupancy() for a in r.get_atoms()]
                ),
                'n_atoms':                 len(list(r.get_atoms())),
                'has_alt_conformation':    int(
                    any(a.is_disordered() for a in r.get_atoms())
                ),
            }

            # Merge all sub-dicts
            for src in [sasa_map, contact_map, hbond_map, sb_map,
                        ss_bond_map, pipi_map, catpi_map, bb_geom_map,
                        bf_map, net_map, wat_map, nbcomp_map]:
                row.update(src.get(key, {}))

            rows.append(row)

        return rows, ss_map


# ─────────────────────────────────────────────────────────────────────────────

class PerAtomExtractor:
    """
    Per-atom feature extraction (~25 features per atom).
    """

    def __init__(self, loader: PDBLoader):
        self.loader    = loader
        self.model     = loader.model
        self.all_atoms = [a for a in self.model.get_atoms()
                          if get_atom_element(a) != 'H']
        self.ns        = NeighborSearch(self.all_atoms)

    def run(self):
        all_coords = np.array([a.get_vector().get_array()
                                for a in self.all_atoms])
        centroid   = all_coords.mean(axis=0)

        rows = []
        for a in self.all_atoms:
            r   = a.get_parent()
            ch  = r.get_parent().id
            elem = get_atom_element(a)
            pos  = a.get_vector().get_array()

            # Neighbours within 3 Å and 5 Å
            nb3 = self.ns.search(pos, 3.0, level='A')
            nb5 = self.ns.search(pos, 5.0, level='A')

            # Distance from protein centroid
            dist_centroid = float(np.linalg.norm(pos - centroid))

            rows.append({
                'chain':       ch,
                'resnum':      r.get_id()[1],
                'resname':     r.get_resname(),
                'atom_serial': a.get_serial_number(),
                'atom_name':   a.get_name().strip(),
                'element':     elem,
                'x':           float(pos[0]),
                'y':           float(pos[1]),
                'z':           float(pos[2]),
                'bfactor':     a.get_bfactor(),
                'occupancy':   a.get_occupancy(),
                'is_backbone': int(a.get_name().strip() in ('N','CA','C','O')),
                'is_sidechain': int(a.get_name().strip() not in
                                    ('N','CA','C','O','OXT')),
                'is_heteroatom': int(r.get_id()[0] != ' '),
                'is_water':    int(r.get_resname() in ('HOH','WAT','H2O')),
                'vdw_radius':  ATOM_VDW_RADIUS.get(elem, 1.70),
                'n_neighbours_3A': len(nb3) - 1,
                'n_neighbours_5A': len(nb5) - 1,
                'dist_from_centroid_A': dist_centroid,
            })
        return rows


# ─────────────────────────────────────────────────────────────────────────────

class ChainFeatureExtractor:
    """
    Per-chain structural features.
    """

    def __init__(self, loader: PDBLoader, per_res_rows: list):
        self.loader       = loader
        self.model        = loader.model
        self.per_res_rows = per_res_rows

    def run(self):
        rows = []
        df   = pd.DataFrame(self.per_res_rows)

        for chain in self.model.get_chains():
            cid   = chain.id
            chain_df = df[df['chain'] == cid] if 'chain' in df.columns else pd.DataFrame()

            aa_res = [r for r in chain.get_residues() if is_aa(r, standard=False)]
            if not aa_res:
                continue

            ca_coords = np.array([
                r['CA'].get_vector().get_array()
                for r in aa_res if 'CA' in r
            ])

            # End-to-end and Rg
            rg, ete = 0.0, 0.0
            if len(ca_coords) > 1:
                c = ca_coords.mean(axis=0)
                rg = float(np.sqrt(np.mean(
                    np.sum((ca_coords - c)**2, axis=1)
                )))
                ete = float(np.linalg.norm(ca_coords[-1] - ca_coords[0]))

            # SS fractions from per-res df
            n_helix = int(chain_df['is_helix'].sum()) if 'is_helix' in chain_df else 0
            n_sheet = int(chain_df['is_sheet'].sum()) if 'is_sheet' in chain_df else 0
            n_coil  = int(chain_df['is_coil'].sum()) if 'is_coil' in chain_df else 0
            total   = len(aa_res)

            row = {
                'chain':           cid,
                'n_residues':      total,
                'n_atoms':         len(list(chain.get_atoms())),
                'radius_of_gyration_A': rg,
                'end_to_end_dist_A': ete,
                'frac_helix':      safe_div(n_helix, total),
                'frac_sheet':      safe_div(n_sheet, total),
                'frac_coil':       safe_div(n_coil, total),
            }

            # Physicochemical chain averages
            for col in ['hydrophobicity_KD','formal_charge','bfactor_res_mean']:
                if col in chain_df.columns:
                    vals = chain_df[col].dropna()
                    row[f'mean_{col}'] = float(vals.mean()) if len(vals) > 0 else None
                    row[f'std_{col}']  = float(vals.std())  if len(vals) > 1 else None

            # SASA chain sum
            if 'sasa_total' in chain_df.columns:
                row['total_sasa_A2'] = float(chain_df['sasa_total'].sum())

            rows.append(row)
        return rows


# ─────────────────────────────────────────────────────────────────────────────

class InteractionFeatureExtractor:
    """
    Collect all pairwise interactions into a dedicated CSV:
      hydrogen bonds, salt bridges, disulfides, pi-pi, cation-pi,
      hydrophobic contacts, and generic CA contacts.
    """

    def __init__(self, loader: PDBLoader):
        self.loader = loader
        self.model  = loader.model
        self.aa_res = [r for chain in self.model
                       for r in chain.get_residues()
                       if is_aa(r, standard=False)]

    def run(self):
        rows = []

        # ── Disulfide bonds ───────────────────────────────────────────────────
        cys_sg = [(r, r['SG']) for r in self.aa_res
                  if r.get_resname() == 'CYS' and 'SG' in r]
        for i, (ri, sgi) in enumerate(cys_sg):
            for j, (rj, sgj) in enumerate(cys_sg):
                if j <= i:
                    continue
                d = (sgi.get_vector() - sgj.get_vector()).norm()
                if d <= SSBOND_DIST:
                    rows.append({
                        'interaction_type': 'disulfide',
                        'chain1': ri.get_parent().id, 'resnum1': ri.get_id()[1],
                        'resname1': 'CYS', 'atom1': 'SG',
                        'chain2': rj.get_parent().id, 'resnum2': rj.get_id()[1],
                        'resname2': 'CYS', 'atom2': 'SG',
                        'distance_A': round(d, 4),
                    })

        # ── Salt bridges ──────────────────────────────────────────────────────
        pos_atoms = {'ARG': ['NH1','NH2'], 'LYS': ['NZ'], 'HIS': ['ND1','NE2']}
        neg_atoms = {'ASP': ['OD1','OD2'], 'GLU': ['OE1','OE2']}
        pos_list, neg_list = [], []
        for r in self.aa_res:
            for aname in pos_atoms.get(r.get_resname(), []):
                if aname in r:
                    pos_list.append((r, r[aname]))
            for aname in neg_atoms.get(r.get_resname(), []):
                if aname in r:
                    neg_list.append((r, r[aname]))

        for pr, pa in pos_list:
            for nr, na in neg_list:
                d = (pa.get_vector() - na.get_vector()).norm()
                if d <= SALTBRIDGE_DIST:
                    rows.append({
                        'interaction_type': 'salt_bridge',
                        'chain1': pr.get_parent().id, 'resnum1': pr.get_id()[1],
                        'resname1': pr.get_resname(), 'atom1': pa.get_name().strip(),
                        'chain2': nr.get_parent().id, 'resnum2': nr.get_id()[1],
                        'resname2': nr.get_resname(), 'atom2': na.get_name().strip(),
                        'distance_A': round(d, 4),
                    })

        # ── Hydrogen bonds ────────────────────────────────────────────────────
        donors    = {'N','NE','NH1','NH2','NZ','ND1','NE2','ND2','NE1',
                     'OG','OG1','OH'}
        acceptors = {'O','OD1','OD2','OE1','OE2','ND1','NE2',
                     'OG','OG1','OH','SD'}
        d_atoms, a_atoms = [], []
        for r in self.aa_res:
            for a in r.get_atoms():
                if a.get_name().strip() in donors:
                    d_atoms.append(a)
                if a.get_name().strip() in acceptors:
                    a_atoms.append(a)

        for da in d_atoms:
            dr = da.get_parent()
            for aa in a_atoms:
                ar = aa.get_parent()
                if dr == ar:
                    continue
                d = (da.get_vector() - aa.get_vector()).norm()
                if d <= HBOND_DA_DIST:
                    rows.append({
                        'interaction_type': 'hydrogen_bond',
                        'chain1': dr.get_parent().id, 'resnum1': dr.get_id()[1],
                        'resname1': dr.get_resname(), 'atom1': da.get_name().strip(),
                        'chain2': ar.get_parent().id, 'resnum2': ar.get_id()[1],
                        'resname2': ar.get_resname(), 'atom2': aa.get_name().strip(),
                        'distance_A': round(d, 4),
                    })

        # ── Pi-pi ─────────────────────────────────────────────────────────────
        aro_res = [r for r in self.aa_res if r.get_resname() in RING_ATOMS]
        for i, ri in enumerate(aro_res):
            ci = ring_centroid(ri, RING_ATOMS[ri.get_resname()])
            ni = ring_normal(ri, RING_ATOMS[ri.get_resname()])
            if ci is None or ni is None:
                continue
            for j, rj in enumerate(aro_res):
                if j <= i:
                    continue
                cj = ring_centroid(rj, RING_ATOMS[rj.get_resname()])
                nj = ring_normal(rj, RING_ATOMS[rj.get_resname()])
                if cj is None or nj is None:
                    continue
                d   = np.linalg.norm(ci - cj)
                ang = vector_angle(ni, nj)
                if d < PIPI_DIST and ang is not None:
                    ang = min(ang, 180-ang)
                    if ang < PIPI_ANGLE:
                        rows.append({
                            'interaction_type': 'pi_pi_stacking',
                            'chain1': ri.get_parent().id,
                            'resnum1': ri.get_id()[1],
                            'resname1': ri.get_resname(), 'atom1': 'ring',
                            'chain2': rj.get_parent().id,
                            'resnum2': rj.get_id()[1],
                            'resname2': rj.get_resname(), 'atom2': 'ring',
                            'distance_A': round(d, 4),
                        })

        # ── Hydrophobic contacts ──────────────────────────────────────────────
        hp_res = [r for r in self.aa_res if r.get_resname() in HYDROPHOBIC_RES]
        ca_hp = {r: r['CA'].get_vector().get_array()
                 for r in hp_res if 'CA' in r}
        seen = set()
        for ri, ci in ca_hp.items():
            for rj, cj in ca_hp.items():
                if ri == rj:
                    continue
                pair = tuple(sorted([id(ri), id(rj)]))
                if pair in seen:
                    continue
                seen.add(pair)
                d = float(np.linalg.norm(ci - cj))
                if d < HYDROPHOBIC_DIST:
                    rows.append({
                        'interaction_type': 'hydrophobic_contact',
                        'chain1': ri.get_parent().id,
                        'resnum1': ri.get_id()[1],
                        'resname1': ri.get_resname(), 'atom1': 'CA',
                        'chain2': rj.get_parent().id,
                        'resnum2': rj.get_id()[1],
                        'resname2': rj.get_resname(), 'atom2': 'CA',
                        'distance_A': round(d, 4),
                    })

        return rows


# ─────────────────────────────────────────────────────────────────────────────

class SecondaryStructureDetailExtractor:
    """
    Extract declared HELIX/SHEET segment data from PDB REMARK records
    and compute per-segment geometry.
    """

    def __init__(self, loader: PDBLoader):
        self.loader = loader
        self.model  = loader.model

    def run(self):
        rows = []

        def get_chain(chain_id):
            """Safely retrieve a chain by ID from the model."""
            try:
                return self.model[chain_id]
            except KeyError:
                return next(iter(self.model.get_chains()), None)

        # Helices
        for h in self.loader.parse_raw_helices():
            chain = get_chain(h['chain'])
            if chain is None:
                rows.append({**h, 'element_type': 'HELIX',
                             'mean_bfactor': None, 'mean_phi': None,
                             'mean_psi': None, 'mean_sasa': None})
                continue

            res_in_seg = [r for r in chain.get_residues()
                          if is_aa(r, standard=False)
                          and h['start'] <= r.get_id()[1] <= h['end']]

            bf = np.mean([a.get_bfactor() for r in res_in_seg
                          for a in r.get_atoms()]) if res_in_seg else None
            rows.append({
                'element_type': 'HELIX',
                'helix_id': h.get('helix_id',''),
                'sheet_id': '',
                'chain': h['chain'],
                'start': h['start'], 'end': h['end'],
                'length': h['end'] - h['start'] + 1,
                'helix_type': h.get('helix_type', ''),
                'sense': '',
                'n_residues_in_segment': len(res_in_seg),
                'mean_bfactor': round(bf, 4) if bf else None,
            })

        # Sheets
        for s in self.loader.parse_raw_sheets():
            chain = get_chain(s['chain'])

            rows.append({
                'element_type': 'SHEET',
                'helix_id': '',
                'sheet_id': s.get('sheet_id',''),
                'chain': s['chain'],
                'start': s['start'], 'end': s['end'],
                'length': s['end'] - s['start'] + 1,
                'helix_type': '',
                'sense': s.get('sense', ''),
                'n_residues_in_segment': s['end'] - s['start'] + 1,
                'mean_bfactor': None,
            })

        return rows


# ─────────────────────────────────────────────────────────────────────────────

class DistanceMatrixExtractor:
    """
    Compute and serialise the full CA-CA distance matrix as a long-format CSV
    (upper triangle only, every pair).
    """

    def __init__(self, loader: PDBLoader):
        self.loader  = loader
        self.model   = loader.model
        self.aa_res  = [r for chain in self.model
                        for r in chain.get_residues()
                        if is_aa(r, standard=False)]

    def run(self):
        entries = [(r.get_parent().id, r.get_id()[1],
                    r.get_resname(),
                    r['CA'].get_vector().get_array())
                   for r in self.aa_res if 'CA' in r]

        rows = []
        n = len(entries)
        for i in range(n):
            for j in range(i+1, n):
                ci, cj = entries[i][3], entries[j][3]
                d = float(np.linalg.norm(ci - cj))
                rows.append({
                    'chain1': entries[i][0], 'resnum1': entries[i][1],
                    'resname1': entries[i][2],
                    'chain2': entries[j][0], 'resnum2': entries[j][1],
                    'resname2': entries[j][2],
                    'ca_dist_A': round(d, 4),
                    'seq_separation': abs(j - i),
                })
        return rows


# ─── PARALLEL HELPER ─────────────────────────────────────────────────────────

def _run_per_residue(loader_args):
    """Top-level function for multiprocessing (must be picklable)."""
    path, verbose = loader_args
    loader  = PDBLoader(path, verbose=False)
    extractor = PerResidueExtractor(loader)
    rows, ss_map = extractor.run()
    return rows, ss_map


# ─── MAIN ORCHESTRATOR ───────────────────────────────────────────────────────

class PDBFeatureFramework:
    """
    Master orchestrator: runs all extractors, writes CSVs.
    """

    OUTPUT_FILES = {
        'global':        '{stem}_global_features.csv',
        'per_residue':   '{stem}_per_residue_features.csv',
        'per_atom':      '{stem}_per_atom_features.csv',
        'per_chain':     '{stem}_per_chain_features.csv',
        'interactions':  '{stem}_interactions.csv',
        'ss_segments':   '{stem}_ss_segments.csv',
        'distance_matrix': '{stem}_distance_matrix.csv',
        'summary':       '{stem}_extraction_summary.txt',
    }

    def __init__(self, pdb_path: str, cores: int = 1,
                 output_prefix: str = None, verbose: bool = True):
        self.pdb_path = str(Path(pdb_path).resolve())
        self.cores    = max(1, cores)
        self.verbose  = verbose
        stem = output_prefix or Path(pdb_path).stem
        self.stems = {k: v.format(stem=stem)
                      for k, v in self.OUTPUT_FILES.items()}

    def run(self):
        t0 = time.time()
        log(f"{'='*60}", self.verbose)
        log(f"  PDB Feature Extractor v{VERSION}", self.verbose)
        log(f"  Input : {self.pdb_path}", self.verbose)
        log(f"  Cores : {self.cores}", self.verbose)
        log(f"{'='*60}", self.verbose)

        # 1. Load PDB
        loader = PDBLoader(self.pdb_path, verbose=self.verbose)

        # 2. Per-residue (core)
        log("Extracting per-residue features ...", self.verbose)
        per_res_ext   = PerResidueExtractor(loader)
        per_res_rows, ss_map = per_res_ext.run()
        log(f"  → {len(per_res_rows)} residue rows", self.verbose)

        # 3. Global
        log("Extracting global features ...", self.verbose)
        global_ext  = GlobalFeatureExtractor(loader)
        global_dict = global_ext.run(ss_map)
        log(f"  → {len(global_dict)} global features", self.verbose)

        # 4. Per-atom
        log("Extracting per-atom features ...", self.verbose)
        atom_ext  = PerAtomExtractor(loader)
        atom_rows = atom_ext.run()
        log(f"  → {len(atom_rows)} atom rows", self.verbose)

        # 5. Per-chain
        log("Extracting per-chain features ...", self.verbose)
        chain_ext  = ChainFeatureExtractor(loader, per_res_rows)
        chain_rows = chain_ext.run()
        log(f"  → {len(chain_rows)} chain rows", self.verbose)

        # 6. Interactions
        log("Extracting pairwise interactions ...", self.verbose)
        inter_ext  = InteractionFeatureExtractor(loader)
        inter_rows = inter_ext.run()
        log(f"  → {len(inter_rows)} interaction pairs", self.verbose)

        # 7. SS segments
        log("Extracting secondary structure segments ...", self.verbose)
        ss_ext  = SecondaryStructureDetailExtractor(loader)
        ss_rows = ss_ext.run()
        log(f"  → {len(ss_rows)} SS segments", self.verbose)

        # 8. Distance matrix
        log("Computing CA distance matrix ...", self.verbose)
        dm_ext  = DistanceMatrixExtractor(loader)
        dm_rows = dm_ext.run()
        log(f"  → {len(dm_rows)} distance pairs", self.verbose)

        # ── Write CSVs ────────────────────────────────────────────────────────
        log("\nWriting output files ...", self.verbose)

        # Global (single-row transposed)
        pd.DataFrame([global_dict]).to_csv(
            self.stems['global'], index=False)
        log(f"  {self.stems['global']}", self.verbose)

        # Per-residue
        pd.DataFrame(per_res_rows).to_csv(
            self.stems['per_residue'], index=False)
        log(f"  {self.stems['per_residue']}", self.verbose)

        # Per-atom
        pd.DataFrame(atom_rows).to_csv(
            self.stems['per_atom'], index=False)
        log(f"  {self.stems['per_atom']}", self.verbose)

        # Per-chain
        pd.DataFrame(chain_rows).to_csv(
            self.stems['per_chain'], index=False)
        log(f"  {self.stems['per_chain']}", self.verbose)

        # Interactions
        pd.DataFrame(inter_rows).to_csv(
            self.stems['interactions'], index=False)
        log(f"  {self.stems['interactions']}", self.verbose)

        # SS segments
        pd.DataFrame(ss_rows).to_csv(
            self.stems['ss_segments'], index=False)
        log(f"  {self.stems['ss_segments']}", self.verbose)

        # Distance matrix
        pd.DataFrame(dm_rows).to_csv(
            self.stems['distance_matrix'], index=False)
        log(f"  {self.stems['distance_matrix']}", self.verbose)

        # Summary report
        elapsed = time.time() - t0
        self._write_summary(loader, global_dict, per_res_rows,
                            atom_rows, inter_rows, elapsed)

        log(f"\n{'='*60}", self.verbose)
        log(f"  Completed in {elapsed:.2f} s", self.verbose)
        log(f"{'='*60}\n", self.verbose)

        return {
            'global':          pd.DataFrame([global_dict]),
            'per_residue':     pd.DataFrame(per_res_rows),
            'per_atom':        pd.DataFrame(atom_rows),
            'per_chain':       pd.DataFrame(chain_rows),
            'interactions':    pd.DataFrame(inter_rows),
            'ss_segments':     pd.DataFrame(ss_rows),
            'distance_matrix': pd.DataFrame(dm_rows),
        }

    def _write_summary(self, loader, gd, res_rows, atom_rows,
                       inter_rows, elapsed):
        lines = [
            f"PDB Feature Extractor v{VERSION} — Extraction Summary",
            "=" * 60,
            f"Input file   : {self.pdb_path}",
            f"PDB type     : {loader.pdb_type}",
            f"PDB ID       : {gd.get('pdb_id','')}",
            f"Title        : {gd.get('title','')}",
            f"Resolution   : {gd.get('resolution_A','')} Å",
            f"Method       : {gd.get('structure_method','')}",
            f"Space group  : {gd.get('space_group','')}",
            "",
            "COMPOSITION",
            f"  Chains       : {gd.get('n_chains','')}",
            f"  Residues     : {gd.get('n_residues','')}",
            f"  Heavy atoms  : {gd.get('n_heavy_atoms','')}",
            f"  Water mols   : {gd.get('n_water','')}",
            f"  HETATM groups: {gd.get('n_hetatm','')}",
            "",
            "SHAPE",
            f"  Rg (CA)      : {gd.get('radius_of_gyration_A', 'N/A'):.3f} Å",
            f"  Max dist     : {gd.get('max_pairwise_dist_A', 'N/A'):.3f} Å",
            f"  MW           : {gd.get('mw_da', 'N/A'):.1f} Da",
            f"  pI (approx)  : {gd.get('pI_approx', 'N/A')}",
            "",
            "SECONDARY STRUCTURE",
            f"  Helix frac   : {gd.get('ss_frac_helix', 0):.3f}",
            f"  Sheet frac   : {gd.get('ss_frac_sheet', 0):.3f}",
            f"  Coil  frac   : {gd.get('ss_frac_coil', 0):.3f}",
            "",
            "OUTPUT FILES",
        ]
        for k, v in self.stems.items():
            lines.append(f"  {k:20s}: {v}")
        lines += [
            "",
            f"Extraction time : {elapsed:.2f} seconds",
            f"Timestamp       : {datetime.now().isoformat()}",
        ]

        with open(self.stems['summary'], 'w') as f:
            f.write('\n'.join(lines))
        log(f"  {self.stems['summary']}", self.verbose)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog='structuraminer.py',
        description='Exhaustive Structural Feature Extraction from PDB files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python structuraminer.py --input 1xei.pdb --cores 4
  python structuraminer.py --input prot.pdb --cores 8 --output my_run
  python structuraminer.py --input prot.pdb --cores 1 --quiet
        """
    )
    parser.add_argument('--input',  '-i', required=True,
                        help='Path to input PDB file')
    parser.add_argument('--cores',  '-c', type=int, default=1,
                        help='Number of CPU cores (default: 1)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file prefix (default: PDB stem name)')
    parser.add_argument('--quiet',  '-q', action='store_true',
                        help='Suppress verbose logging')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {VERSION}')
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    framework = PDBFeatureFramework(
        pdb_path      = args.input,
        cores         = args.cores,
        output_prefix = args.output,
        verbose       = not args.quiet,
    )
    framework.run()


if __name__ == '__main__':
    main()
