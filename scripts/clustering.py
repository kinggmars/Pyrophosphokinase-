#!/usr/bin/env python3
"""
Cluster PDB models by pairwise RMSD, then report representative conformations.

Example:
    python scripts/clustering.py --pdb-glob "data/predictions/af-cluster/pdb/*.pdb" \
        --eps 2.0 --min-samples 2 --top-k 3 --plot rmsd.png
"""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from sklearn.cluster import DBSCAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DBSCAN clustering on PDB models using pairwise RMSD."
    )
    parser.add_argument(
        "--pdb-glob",
        default="data/predictions/af-cluster/pdb/*.pdb",
        help="Glob for input PDB files (sorted lexicographically).",
    )
    parser.add_argument(
        "--pdb-list",
        type=Path,
        help="Optional text file with one PDB path per line (overrides --pdb-glob).",
    )
    parser.add_argument(
        "--atom-selection",
        default="name CA",
        help="MDTraj selection string; default uses only CA atoms for alignment.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=2.0,
        help="DBSCAN eps (in chosen RMSD unit).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples per cluster for DBSCAN.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of largest clusters to report as major conformations.",
    )
    parser.add_argument(
        "--unit",
        choices=["angstrom", "nm"],
        default="angstrom",
        help="Unit for RMSD output and eps (mdtraj returns nm; angstrom is scaled).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional path to save an RMSD heatmap with cluster ordering (png/pdf).",
    )
    parser.add_argument(
        "--save-matrix",
        type=Path,
        help="Optional path to save the RMSD matrix as .npy.",
    )
    return parser.parse_args()


def read_pdb_paths(pdb_glob: str, pdb_list: Path | None) -> List[Path]:
    if pdb_list:
        paths = [
            Path(line.strip()).expanduser()
            for line in pdb_list.read_text().splitlines()
            if line.strip()
        ]
    else:
        paths = [Path(p).expanduser() for p in sorted(glob.glob(pdb_glob))]
    if not paths:
        raise ValueError("No PDB files found; check --pdb-glob/--pdb-list.")
    return paths


def load_structures(paths: Sequence[Path], atom_selection: str) -> List[md.Trajectory]:
    trajectories: List[md.Trajectory] = []
    n_atoms: int | None = None
    for path in paths:
        traj = md.load(str(path))
        atom_idx = traj.topology.select(atom_selection)
        if atom_idx.size == 0:
            raise ValueError(f"No atoms match selection '{atom_selection}' in {path}")
        traj = traj.atom_slice(atom_idx)
        if n_atoms is None:
            n_atoms = traj.n_atoms
        elif traj.n_atoms != n_atoms:
            raise ValueError(
                f"Atom count mismatch after selection: {path} has {traj.n_atoms}, "
                f"expected {n_atoms}."
            )
        trajectories.append(traj)
    return trajectories


def compute_rmsd_matrix(trajs: Sequence[md.Trajectory]) -> np.ndarray:
    n = len(trajs)
    rmsd = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            val = float(md.rmsd(trajs[j], trajs[i])[0])  # mdtraj returns nm
            rmsd[i, j] = val
            rmsd[j, i] = val
    return rmsd


def identify_major_clusters(
    labels: np.ndarray, rmsd: np.ndarray, top_k: int
) -> Tuple[List[Tuple[int, List[int]]], dict[int, int]]:
    cluster_members: dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        cluster_members.setdefault(label, []).append(idx)
    if not cluster_members:
        return [], {}
    ordered = sorted(cluster_members.items(), key=lambda kv: len(kv[1]), reverse=True)
    major = ordered[:top_k]
    medoids: dict[int, int] = {}
    for cluster_id, indices in major:
        sub = rmsd[np.ix_(indices, indices)]
        medoid_local = int(np.argmin(sub.sum(axis=1)))
        medoids[cluster_id] = indices[medoid_local]
    return major, medoids


def reorder_by_cluster(labels: np.ndarray) -> List[int]:
    # Noise (-1) sent to the end.
    return sorted(range(len(labels)), key=lambda i: (labels[i], i))


def plot_rmsd_heatmap(
    rmsd: np.ndarray,
    labels: np.ndarray,
    paths: Sequence[Path],
    outfile: Path,
    unit: str,
) -> None:
    order = reorder_by_cluster(labels)
    ordered_rmsd = rmsd[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ordered_rmsd, cmap="magma", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"RMSD ({unit})")
    ax.set_title("RMSD clustering (DBSCAN)")
    ax.set_xlabel("Model (ordered by cluster)")
    ax.set_ylabel("Model (ordered by cluster)")
    step = max(1, len(paths) // 10)
    ticks = list(range(0, len(paths), step))
    labels_ordered = [paths[i].stem for i in order]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels_ordered[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([labels_ordered[i] for i in ticks], fontsize=7)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pdb_paths = read_pdb_paths(args.pdb_glob, args.pdb_list)
    print(f"Found {len(pdb_paths)} PDBs.")

    trajectories = load_structures(pdb_paths, args.atom_selection)
    rmsd_nm = compute_rmsd_matrix(trajectories)
    if args.unit == "angstrom":
        rmsd = rmsd_nm * 10.0
        unit_label = "Å"
    else:
        rmsd = rmsd_nm
        unit_label = "nm"

    clustering = DBSCAN(
        eps=args.eps, min_samples=args.min_samples, metric="precomputed"
    ).fit(rmsd)
    labels = clustering.labels_

    counts = Counter(labels)
    print("Cluster counts (label: size):")
    for label, size in sorted(counts.items(), key=lambda kv: kv[0]):
        label_str = "noise" if label == -1 else str(label)
        print(f"  {label_str}: {size}")

    major, medoids = identify_major_clusters(labels, rmsd, args.top_k)
    if not major:
        print("No clusters found (all points labeled as noise).")
    else:
        print(f"Top {len(major)} clusters and representatives:")
        for cluster_id, members in major:
            medoid_idx = medoids[cluster_id]
            medoid_name = pdb_paths[medoid_idx].name
            print(
                f"  Cluster {cluster_id}: size={len(members)}, "
                f"representative={medoid_name} (index {medoid_idx})"
            )

    if args.save_matrix:
        np.save(args.save_matrix, rmsd)
        print(f"Saved RMSD matrix to {args.save_matrix}")

    if args.plot:
        plot_rmsd_heatmap(rmsd, labels, pdb_paths, args.plot, unit_label)
        print(f"Saved heatmap to {args.plot}")


if __name__ == "__main__":
    main()
