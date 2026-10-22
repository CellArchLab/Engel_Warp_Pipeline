# [Engel Lab](https://www.cellarchlab.com/) Warp Pipeline

A compact workflow connecting [Warp/M](https://github.com/warpem/warp), [Miss-Alignment](https://github.com/warpem/miss-alignment), [pytom-match-pick](https://github.com/SBC-Utrecht/pytom-match-pick), and [IsoNet2](https://github.com/IsoNet-cryoET/IsoNet2).

---

## Pipeline Overview

| Step | Script | Description |
|:---:|---|---|
| 1 | `submit_warp-missalignment.sh` | Processes raw tilt-series data in Warp, performs initial IMOD or AreTomo alignment, runs Miss-Alignment, and reconstructs the final tomograms in `warp_tiltseries_<RUN_NAME>/reconstruction_miss/` |
| 2 | `warp2pytom.py` | Submits PyTom template-matching jobs from Warp XML metadata, then extracts particle candidates in a separate SLURM array workflow |
| 3 | `submit_export_particles.sh` | Merges pytom particle STAR files, normalizes their tomogram identifiers for WarpTools, exports particles, and merges the resulting metadata across datasets for [RELION5](https://github.com/3dem/relion/tree/ver5.0) |
| 4 | `isonet2_pipeline` | Trains and applies IsoNet2 using the post-Miss-Alignment odd/even half tomograms, tilt angles, Warp XML defocus values, and user-supplied masks |

---

## 1. `submit_warp-missalignment.sh` — Warp/M and Miss-Alignment

This script submits a full Warp pipline as well as MissAlignment automatically. All possible variables can be edited in a header block of the shell script.

```bash
sbatch submit_warp-missalignment.sh
```

---

## 2. `warp2pytom.py` — PyTom Template Matching and Candidate Extraction

The wrapper provides two workflows: `--workflow match` (default) and `--workflow extract`. Run matching first, then extract candidates from the generated `*_job.json` files.

### Template matching (`--workflow match`)

Check `--help` for all flag options and description especially for the SLURM header.
This script automates batch submission of `pytom-match-pick` jobs on an HPC cluster (SLURM) by reading metadata directly from Warp tilt-series XMLs. It:

- Extracts **tilt angles** from `<Angles>` in each `*.xml` (sign-flipped to the pytom convention)
- Reads **per-tilt defocus** (μm) from the `<GridCTF>`
- Builds **per-tilt exposure** from `<Dose>`
- Locates the matching reconstruction in `reconstruction_miss/`

It runs on all tilt-series matched by `--pattern` (default `*.xml`) unless restricted with `--include` / `--exclude`.

`--dry-run` generates the bash scripts without submitting them, allowing for quick sanity checks or manual execution.

**Submission modes:**

| Mode | Behavior |
|---|---|
| `array` *(default)* | Single SLURM **array job** (`submit_array.sh`) over all tomograms |
| `per-tomo` | One standalone `submit_<prefix>.sh` per tomogram |

### Matching example

```bash
python warp2pytom.py \
  --workflow match \
  -i /path/to/warp/tiltseries \
  -d submission \
  -t /path/to/template.mrc \
  -m /path/to/mask.mrc \
  -g 0 \
  --voxel-size-angstrom 10 \
  --angular-search 10 \
  --mode array \
  --per-tilt-weighting \
  --tomogram-ctf-model phase-flip \
  --dry-run
```

Use `--dose 2` as a fallback when Warp XML has no `<Dose>` values. You can also pass `--bmask-dir`, `-s 2 2 1`, `--non-spherical-mask`, `-r`, `--rng-seed`, and `--include`/`--exclude` filters. The matching workflow requires `--warp-dir`, `--template`, `--mask`, `--gpu-ids`, `--voxel-size-angstrom`, and **exactly one** of `--particle-diameter` or `--angular-search`.

`--mode array` creates `submission/submit_array.sh` (default); `--mode per-tomo` creates one submission script per tomogram. This setting applies only to matching; extraction always uses an array.

**Defaults written automatically:**

```
--amplitude-contrast 0.07
--spherical-aberration 2.7
--voltage 300
```

### Candidate extraction (`--workflow extract`)

After template matching finishes, submit particle-candidate extraction directly through the same wrapper. **No Warp XML directory, template, mask, GPU IDs, or voxel size is needed for this workflow.** The wrapper recursively discovers `*_job.json` files under `--output-dir`, sorts them lexicographically, and creates one SLURM array task per JSON.

```bash
python warp2pytom.py \
  --workflow extract \
  -d submission/
```

The default command for each array task is equivalent to:

```bash
pytom_extract_candidates.py \
  -j submission/Position_1/Position_1_job.json \
  --particle-diameter 300 \
  -n 1000 \
  -c 0.1
```

**`--relion5-compat` is OFF by default** for WarpTools-compatible output. Supply `--relion5-compat` only when RELION5-compatible particle output is desired.

| Extraction option | Default | Purpose |
|---|---|---|
| `--particle-diameter` | `300` Å | Extraction particle diameter (independent of the matching search setting) |
| `-n`, `--n-candidates` | `1000` | Maximum number of candidate particles per tomogram |
| `-c`, `--cutoff` | `0.1` | Minimum score threshold |
| `--relion5-compat` | off | Opt in to RELION5-compatible output; leave off for WarpTools |
| `--extract-extra-args` | empty | Other extraction CLI flags, as a single shell-quoted string |
| `--array-max-parallel`, `--max-parallel` | `20` | Maximum concurrent SLURM array tasks |

To select subsets or customize extraction:

```bash
python warp2pytom.py \
  --workflow extract \
  -d submission/ \
  --include 'Position_1*' \
  --particle-diameter 250 \
  -n 2000 \
  -c 0.15 \
  --extract-extra-args '--some-supported-option value' \
  --dry-run
```

Use `--include`/`--exclude` to filter using a job JSON's enclosing folder name or basename (wildcards supported). Arguments managed directly by the wrapper (for example `-j`, `-n`, `-c`, and `--particle-diameter`) cannot be overridden through `--extract-extra-args`.

**GPU enforcement:** each extraction task requests `#SBATCH --gres=gpu:1`, even if a different `--gres` value is supplied. Other resource settings use the standard SLURM options; defaults include partition `emgpu`, QoS `emgpu,emgpu-fast`, 120 GB memory, 4 CPUs per task, and 5 hours.

**Generated extraction files:**

```text
submission/
├── Position_1/
│   └── Position_1_job.json
├── Position_2/
│   └── Position_2_job.json
├── extract_jobs.txt
├── submit_extract_array.sh
└── logs/
    ├── pytom_extract_<jobid>_0.out
    └── pytom_extract_<jobid>_1.out
```

The log directory `submission/logs/` is also used for template-matching SLURM logs. Candidate STAR files are produced by PyTom in the individual matching output directories. `--dry-run` generates the array script and manifest without calling `sbatch`.

---

## 3. `submit_export_particles.sh` — Warp Particle Export

After completing `warp2pytom.py --workflow extract`, the `*_particles.star` files in the individual `submission/<prefix>/` folders can be used for WarpTools particle export with `submit_export_particles.sh`. Keep `--relion5-compat` disabled during candidate extraction for this WarpTools workflow.

Set one entry per dataset in `DATASET_TAGS`, `PYTOM_DIRS`, and `WARP_SETTINGS`. Each pytom directory is expected to contain particle STAR files under `<PYTOM_DIR>/*/*.star`, as produced after candidate extraction from the default `warp2pytom.py` submission structure.

| `EXPORT_DIM` | Output |
|---|---|
| `"2d"` | Per-tilt particle series; writes merged RELION5 `--tomo` particle, tomogram, and optimisation-set STAR files |
| `"3d"` | Subtomograms; writes one merged conventional RELION particle STAR file |

The input STAR files must contain `rlnCoordinateX/Y/Z` and `rlnMicrographName`.

- `COORDS_ANGPIX` — coordinate pixel size
- `OUTPUT_ANGPIX` — requested particle pixel size
- `DIAMETER_ANGSTROM` — diameter, in Å

```bash
sbatch submit_export_particles.sh
```

> The merge step requires Python with `pandas` and `starfile`, normally available in the pytom-match-pick environment.

---

## 4. `isonet2_pipeline` — IsoNet2 Denoising

The `isonet2_pipeline` script facilitates submitting jobs, taking care of preparing the necessary config files and scripts. The default is to run training and prediction, but prediction-only using a pre-trained model is also supported.

Type `isonet2_pipeline --help` to see all options.

### Inputs

* Folders containing odd/even tomogram reconstructions (e.g. generated by AreTomo3 or WarpTools)
* Folder containing tilt angles supplied as .tlt (IMOD), .aln (AreTomo3) or .xml (WarpTools) files per tomogram
  * Alternatively, supply min/max tilt angles to be used for all tomograms
* Folder containing boundary masks (optional)
* Folder containing defocus information (CTF files), supported formats are:
  * `_CTF.txt` files from AreTomo3
  * `.xml files` from WarpTools

### Example usage

```bash
isonet2_pipeline \
    --even-dir even/ \
    --odd-dir odd/ \
    --xml-dir warp_xml/ \
    --bmask-dir bmasks/ \
    --save-dir isonet2_output/
```
---

## License

MIT
