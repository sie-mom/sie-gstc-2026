#!/usr/bin/env bash
#SBATCH --job-name=omega-comsol-6.1
#SBATCH --account=jordan_budhu
#SBATCH --partition=normal_q
#SBATCH --qos=tc_normal_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-comsol-6.1-%j.out
#SBATCH --error=slurm-comsol-6.1-%j.out

set -euo pipefail

# Specific license server for Jordan Budhu group COMSOL 6.1 (1 RF seat)
export LM_LICENSE_FILE="1718@128.173.92.65"

project_root="${SIE_MOM_ROOT:-/home/germanf/repo/sie_mom}"
stage="${COMSOL_STAGE:-omega_sphere}"
case_id="${COMSOL_CASE_ID:-budhu_10ghz_eps7}"
run_name="${COMSOL_RUN_NAME:-${stage}-6.1-${SLURM_JOB_ID}}"
run_dir="$project_root/applications/omega_biisotropic_sphere/results"
run_dir="$run_dir/$case_id/comsol/$run_name"

mkdir -p "$run_dir/logs"
exec > >(tee -a "$run_dir/logs/slurm-${SLURM_JOB_ID}.log") 2>&1

echo "COMSOL 6.1 job ${SLURM_JOB_ID} started on $(hostname) at $(date --iso-8601=seconds)"

module reset
module use /projects/jordan_budhu/modules/tinkercliffs-rome/all
module load comsol/6.1

cd "$project_root"
source .venv/bin/activate

mesh_preset="${COMSOL_MESH_PRESET:-4}"
outer_radius="${COMSOL_OUTER_RADIUS:-1.8}"
zx_grid_size="${COMSOL_ZX_GRID_SIZE:-300}"
radial_theta="${COMSOL_RADIAL_THETA:-45.0}"
radial_phi="${COMSOL_RADIAL_PHI:-0.0}"
radial_points="${COMSOL_RADIAL_POINTS:-501}"
air_extent="${COMSOL_AIR_EXTENT:-0.95}"
boundary_divisor="${COMSOL_BOUNDARY_DIVISOR:-30.0}"
linear_solver="${COMSOL_LINEAR_SOLVER:-direct}"

module list
python -u applications/omega_biisotropic_sphere/comsol/dipole_impedance_matching.py \
    --stage "$stage" \
    --case-id "$case_id" \
    --run-name "$run_name" \
    --run-dir "$run_dir" \
    --version "6.1" \
    --physics-interface "emw" \
    --cores "$SLURM_CPUS_PER_TASK" \
    --linear-solver "$linear_solver" \
    --mesh-preset "$mesh_preset" \
    --boundary-hmax-divisor "$boundary_divisor" \
    --outer-radius "$outer_radius" \
    --air-extent "$air_extent" \
    --radial-theta "$radial_theta" \
    --radial-phi "$radial_phi" \
    --radial-points "$radial_points" \
    --zx-grid-size "$zx_grid_size"

python -c 'import json, pathlib, sys; manifest = json.loads(pathlib.Path(sys.argv[1]).read_text()); sys.exit(0 if manifest.get("status") == "complete" else 1)' "$run_dir/run.json"

echo "COMSOL 6.1 job ${SLURM_JOB_ID} finished at $(date --iso-8601=seconds)"

