#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Create PyTom MatchPick submission from Warp tilt-series XMLs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--warp-dir", required=True, help="Warp tilt-series directory containing Position_*.xml and reconstruction/")
    p.add_argument("-d", "--output-dir", default="submission", help="Output directory")
    p.add_argument("--pattern", default="*.xml", help="XML glob pattern inside --warp-dir")
    p.add_argument("--bmask-dir", help="Optional tomogram-mask directory with files named <prefix>.mrc")
    p.add_argument("--mode", choices=["array", "per-tomo"], default="array")

    p.add_argument("-t", "--template", required=True)
    p.add_argument("-m", "--mask", required=True)
    p.add_argument("-g", "--gpu-ids", nargs="+", required=True)
    p.add_argument("--voxel-size-angstrom", type=float, required=True)
    p.add_argument("--dose", type=float, help="Fallback per-tilt dose if XML has no <Dose>")
    p.add_argument("--include", nargs="+")
    p.add_argument("--exclude", nargs="+")
    p.add_argument("--dry-run", action="store_true")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--particle-diameter", type=float)
    grp.add_argument("--angular-search")

    p.add_argument("--non-spherical-mask", action="store_true")
    p.add_argument("--z-axis-rotational-symmetry", type=int)
    p.add_argument("-s", "--volume-split", nargs=3, type=int, metavar=("X", "Y", "Z"))
    p.add_argument("--search-x", nargs=2, type=int, metavar=("START", "END"))
    p.add_argument("--search-y", nargs=2, type=int, metavar=("START", "END"))
    p.add_argument("--search-z", nargs=2, type=int, metavar=("START", "END"))
    p.add_argument("--tomogram-ctf-model", choices=["phase-flip"])
    p.add_argument("-r", "--random-phase-correction", action="store_true")
    p.add_argument("--half-precision", action="store_true")
    p.add_argument("--rng-seed", type=int, default=69)
    p.add_argument("--per-tilt-weighting", action="store_true")
    p.add_argument("--low-pass", type=float)
    p.add_argument("--high-pass", type=float)

    p.add_argument("--amplitude-contrast", type=float, default=0.07)
    p.add_argument("--spherical-aberration", type=float, default=2.7)
    p.add_argument("--voltage", type=float, default=300.0)
    p.add_argument("--phase-shift", type=float, help="Only written if > 0")
    p.add_argument("--defocus-handedness", type=int, choices=[-1, 0, 1])
    p.add_argument("--spectral-whitening", action="store_true")
    p.add_argument("--log", choices=["info", "debug"])

    p.add_argument("--partition", default="emgpu")
    p.add_argument("--ntasks", default="1")
    p.add_argument("--nodes", default="1")
    p.add_argument("--ntasks-per-node", default="1")
    p.add_argument("--cpus-per-task", default="4")
    p.add_argument("--gres", default="gpu:1")
    p.add_argument("--mem", default="128")
    p.add_argument("--qos", default="emgpu")
    p.add_argument("--time", default="05:00:00")
    p.add_argument("--mail-type", default="none")
    p.add_argument("--array-max-parallel", type=int, default=8)
    p.add_argument("--exclude-nodes", nargs="+")
    p.add_argument("--include-nodes", nargs="+")
    return p.parse_args()


def wanted(prefix, include, exclude):
    if include and not any(fnmatch.fnmatch(prefix, pat) for pat in include):
        return False
    if exclude and any(fnmatch.fnmatch(prefix, pat) for pat in exclude):
        return False
    return True


def flatten(nodes):
    out = []
    for node in nodes:
        for x in (node.text or "").splitlines():
            x = x.strip()
            if x:
                out.append(float(x))
    return out


def parse_warp_xml(xml_file: Path, fallback_dose):
    root = ET.parse(xml_file).getroot()

    angles = [-x for x in flatten(root.findall(".//Angles"))]
    dose = flatten(root.findall(".//Dose"))

    defocus_nodes = root.findall(".//GridCTF/Node") or root.findall(".//GridCTFDefocus/Node")
    defocus = [float(n.attrib["Value"]) for n in defocus_nodes]

    if not angles:
        raise ValueError(f"No tilt angles found in {xml_file}")
    if not defocus:
        raise ValueError(f"No defocus values found in {xml_file}")
    if len(defocus) != len(angles):
        raise ValueError(f"{xml_file}: {len(defocus)} defocus values but {len(angles)} tilt angles")

    if dose:
        if len(dose) != len(angles):
            raise ValueError(f"{xml_file}: {len(dose)} dose values but {len(angles)} tilt angles")
    elif fallback_dose is not None:
        dose = [i * fallback_dose for i in range(len(angles))]
    else:
        raise ValueError(f"{xml_file}: no <Dose> entries found; provide --dose fallback")

    return angles, defocus, dose


def belongs_to_other(stem: str, prefix: str, prefixes) -> bool:
    return any(q != prefix and q.startswith(f"{prefix}_") and (stem == q or stem.startswith(f"{q}_")) for q in prefixes)


def find_tomogram(recon: Path, prefix: str, prefixes) -> Path:
    candidates = sorted(recon.glob(f"{prefix}*.mrc"))
    hits = []

    for p in candidates:
        stem = p.stem

        # Plain exact tomogram: Position_15.mrc
        norm = stem

        # Warp reconstruction: Position_15_9.68Apx.mrc, Position_15_1.5Apx.mrc, etc.
        if norm.endswith("Apx"):
            parts = norm.rsplit("_", 1)
            if len(parts) == 2 and parts[1].endswith("Apx"):
                value = parts[1][:-3]
                try:
                    float(value)
                    norm = parts[0]
                except ValueError:
                    pass

        # Alternative convention: Position_15_Vol.mrc
        if norm.endswith("_Vol"):
            norm = norm[:-4]

        if norm == prefix:
            hits.append(p)

    if not hits:
        raise FileNotFoundError(f"Could not find tomogram for {prefix} in {recon}")

    if len(hits) > 1:
        raise RuntimeError(f"Ambiguous tomogram match for {prefix}: {hits}")

    return hits[0]


def write_list(path: Path, values):
    path.write_text("".join(f"{v}\n" for v in values))


def resolved(path):
    return str(Path(path).resolve())


def bmask_dir(args):
    return resolved(args.bmask_dir) if args.bmask_dir else ""


def sbatch_header(args, output_dir, log_line, job_name, array_spec=None):
    lines = [
        "#!/bin/bash -l",
        f"#SBATCH -D {output_dir}",
        log_line,
        f"#SBATCH -J {job_name}",
        f"#SBATCH --partition={args.partition}",
    ]
    if array_spec:
        lines.append(f"#SBATCH --array={array_spec}")
    lines += [
        f"#SBATCH --ntasks={args.ntasks}",
        f"#SBATCH --nodes={args.nodes}",
        f"#SBATCH --ntasks-per-node={args.ntasks_per_node}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --gres={args.gres}",
        f"#SBATCH --mail-type={args.mail_type}",
        f"#SBATCH --mem={args.mem}G",
        f"#SBATCH --qos={args.qos}",
        f"#SBATCH --time={args.time}",
    ]
    if args.exclude_nodes:
        lines.append(f"#SBATCH --exclude={','.join(args.exclude_nodes)}")
    if args.include_nodes:
        lines.append(f"#SBATCH --nodelist={','.join(args.include_nodes)}")
    lines += ["", "ml purge", "ml pytom-match-pick", ""]
    return "\n".join(lines) + "\n"


def pytom_command(args):
    parts = [
        "pytom_match_template.py",
        '-v "${TOMO}"',
        '-a "${TLT}"',
        '--dose-accumulation "${EXP}"',
        '--defocus "${DF}"',
        '-t "${TEMPLATE}"',
        '-d "${OD}"',
        '-m "${PMASK}"',
    ]
    if args.particle_diameter is not None:
        parts.append(f"--particle-diameter {args.particle_diameter}")
    if args.angular_search is not None:
        parts.append(f"--angular-search {args.angular_search}")
    if args.z_axis_rotational_symmetry is not None:
        parts.append(f"--z-axis-rotational-symmetry {args.z_axis_rotational_symmetry}")
    if args.volume_split:
        parts.append(f"-s {' '.join(map(str, args.volume_split))}")
    if args.search_x:
        parts.append(f"--search-x {args.search_x[0]} {args.search_x[1]}")
    if args.search_y:
        parts.append(f"--search-y {args.search_y[0]} {args.search_y[1]}")
    if args.search_z:
        parts.append(f"--search-z {args.search_z[0]} {args.search_z[1]}")
    parts.append(f"--voxel-size-angstrom {args.voxel_size_angstrom}")
    if args.low_pass is not None:
        parts.append(f"--low-pass {args.low_pass}")
    if args.high_pass is not None:
        parts.append(f"--high-pass {args.high_pass}")
    if args.random_phase_correction:
        parts += ["-r", f"--rng-seed {args.rng_seed}"]
    parts.append(f"-g {' '.join(args.gpu_ids)}")
    if args.per_tilt_weighting:
        parts.append("--per-tilt-weighting")
    if args.tomogram_ctf_model is not None:
        parts.append(f"--tomogram-ctf-model {args.tomogram_ctf_model}")
    if args.non_spherical_mask:
        parts.append("--non-spherical-mask")
    if args.spectral_whitening:
        parts.append("--spectral-whitening")
    if args.half_precision:
        parts.append("--half-precision")
    if args.defocus_handedness is not None:
        parts.append(f"--defocus-handedness {args.defocus_handedness}")
    if args.log is not None:
        parts.append(f"--log {args.log}")
    parts.append(f"--amplitude-contrast {args.amplitude_contrast}")
    parts.append(f"--spherical-aberration {args.spherical_aberration}")
    parts.append(f"--voltage {args.voltage}")
    parts += ["${PHASE_ARGS}", "${TOMO_MASK_ARGS}"]
    return " \\\n  ".join(parts) + "\n"


def job_body(args):
    lines = [
        'TLT="${OD}/${PREFIX}.tlt"',
        'DF="${OD}/${PREFIX}_defocus.txt"',
        'EXP="${OD}/${PREFIX}_exposure.txt"',
        'if [[ -z "${TOMO}" || ! -f "${TOMO}" ]]; then echo "Tomogram not found for ${PREFIX}"; exit 2; fi',
        'TOMO_MASK_ARGS=""',
        'if [[ -n "${BMASK_DIR}" && -f "${BMASK_DIR}/${PREFIX}.mrc" ]]; then',
        '  TOMO_MASK_ARGS="--tomogram-mask ${BMASK_DIR}/${PREFIX}.mrc"',
        "fi",
        'PHASE_ARGS=""',
    ]
    if args.phase_shift is not None and args.phase_shift > 0:
        lines.append(f'PHASE_ARGS="--phase-shift {args.phase_shift}"')
    return "\n".join(lines) + "\n\n" + pytom_command(args)


def make_array_sbatch(jobs, args, output_dir: Path):
    prefixes = [p for p, _ in jobs]
    prefix_list = output_dir / "prefixes.txt"
    tomo_list = output_dir / "tomograms.txt"
    prefix_list.write_text("".join(f"{p}\n" for p in prefixes))
    tomo_list.write_text("".join(f"{t}\n" for _, t in jobs))

    array_spec = f"0-{len(prefixes) - 1}"
    if args.array_max_parallel:
        array_spec += f"%{args.array_max_parallel}"

    script = output_dir / "submit_array.sh"
    with open(script, "w") as f:
        f.write(sbatch_header(args, output_dir, "#SBATCH -o pytom_%A_%a.out\n##SBATCH -e pytom_%A_%a.err", "pytom_tm", array_spec))
        f.write(f'PREFIX_LIST="{prefix_list}"\n')
        f.write(f'TOMO_LIST="{tomo_list}"\n')
        f.write(f'OUT_DIR="{output_dir}"\n')
        f.write(f'TEMPLATE="{resolved(args.template)}"\n')
        f.write(f'PMASK="{resolved(args.mask)}"\n')
        f.write(f'BMASK_DIR="{bmask_dir(args)}"\n')
        f.write('PREFIX=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${PREFIX_LIST}")\n')
        f.write('TOMO=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${TOMO_LIST}")\n')
        f.write('if [[ -z "${PREFIX}" ]]; then echo "Empty prefix for task ${SLURM_ARRAY_TASK_ID}"; exit 2; fi\n')
        f.write('OD="${OUT_DIR}/${PREFIX}"\n')
        f.write('mkdir -p "${OD}"\n')
        f.write(job_body(args))
    os.chmod(script, 0o755)
    return script


def make_per_tomo(prefix, tomo, args, output_dir: Path):
    od = output_dir / prefix
    script = od / f"submit_{prefix}.sh"
    with open(script, "w") as f:
        f.write(sbatch_header(args, output_dir, "#SBATCH -o pytom.out%j", f"pytom_{prefix.split('_')[-1]}"))
        f.write(f'TEMPLATE="{resolved(args.template)}"\n')
        f.write(f'PMASK="{resolved(args.mask)}"\n')
        f.write(f'BMASK_DIR="{bmask_dir(args)}"\n')
        f.write(f'PREFIX="{prefix}"\n')
        f.write(f'OD="{od}"\n')
        f.write(f'TOMO="{tomo}"\n')
        f.write(job_body(args))
    os.chmod(script, 0o755)
    return script


def submit(script, dry_run):
    if dry_run:
        print(f"[dry-run] sbatch {script}")
    else:
        subprocess.run(["sbatch", str(script)], check=False)


def main():
    args = parse_args()
    warp_dir = Path(args.warp_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    recon = warp_dir / "reconstruction_miss"
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = [x for x in sorted(warp_dir.glob(args.pattern)) if wanted(x.stem, args.include, args.exclude)]
    if not xml_files:
        raise SystemExit("No matching XML files found")

    prefixes = {x.stem for x in xml_files}
    jobs = []

    for xml in xml_files:
        prefix = xml.stem
        angles, defocus, dose = parse_warp_xml(xml, args.dose)
        tomo = find_tomogram(recon, prefix, prefixes)
        od = output_dir / prefix
        od.mkdir(parents=True, exist_ok=True)

        write_list(od / f"{prefix}.tlt", angles)
        write_list(od / f"{prefix}_defocus.txt", defocus)
        write_list(od / f"{prefix}_exposure.txt", dose)

        jobs.append((prefix, str(tomo)))

    if args.mode == "array":
        submit(make_array_sbatch(jobs, args, output_dir), args.dry_run)
    else:
        for prefix, tomo in jobs:
            submit(make_per_tomo(prefix, tomo, args, output_dir), args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
