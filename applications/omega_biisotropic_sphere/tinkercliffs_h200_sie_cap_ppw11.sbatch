#!/bin/bash
#SBATCH --job-name=omega-cap11
#SBATCH --account=jordan_budhu
#SBATCH --partition=h200_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-omega-cap11-%j.out
#SBATCH --error=slurm-omega-cap11-%j.err

set -euo pipefail

module reset
module load CMake
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.0.2

repository="${SIE_MOM_REPO:-/home/germanf/repo/sie_mom}"
configuration="applications/omega_biisotropic_sphere/configurations/budhu_10ghz_eps7_cap5_ppw11.json"
run_directory="applications/omega_biisotropic_sphere/results/budhu_10ghz_eps7_cap5_ppw11/sie/least_squares"
build_directory="build-tinkercliffs-sm90"
executable="${build_directory}/applications/omega_biisotropic_sphere/omega_sphere_sie"

if [[ ! -d "${repository}/.git" ]]; then
    echo "Repository not found at ${repository}" >&2
    exit 2
fi
cd "${repository}"

if [[ ! -f .venv/bin/activate ]]; then
    echo "Python environment not found at ${repository}/.venv" >&2
    exit 2
fi
source .venv/bin/activate

if [[ ! -f "${configuration}" ]]; then
    echo "Missing configuration: ${configuration}" >&2
    exit 2
fi

if [[ "${FORCE_RERUN:-0}" != "1" \
      && -f "${run_directory}/run.json" \
      && -f "${run_directory}/solver_metrics.json" \
      && -f "${run_directory}/fields/sie_physical_zx.csv" ]]; then
    echo "PPW 11 already appears complete at ${run_directory}"
    echo "Submit with --export=ALL,FORCE_RERUN=1 to run it again."
    exit 0
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${run_directory}"

echo "Omega-cap SIE H200 PPW 11 run"
echo "  started:       $(date --iso-8601=seconds)"
echo "  host:          $(hostname)"
echo "  job:           ${SLURM_JOB_ID}"
echo "  repository:    ${repository}"
echo "  git commit:    $(git rev-parse HEAD)"
echo "  configuration: ${configuration}"
echo "  result:        ${run_directory}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

{
    echo "started=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "configuration=${configuration}"
    echo "cuda_architecture=90"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} > "${run_directory}/slurm_environment.txt"

cmake -S . -B "${build_directory}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSIE_MOM_USE_DOUBLE_PRECISION=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "${build_directory}" \
    --target omega_sphere_sie \
    --parallel "${SLURM_CPUS_PER_TASK}"

if [[ ! -x "${executable}" ]]; then
    echo "Expected executable was not produced: ${executable}" >&2
    exit 2
fi

srun --ntasks=1 --cpu-bind=cores \
    .venv/bin/python \
    applications/omega_biisotropic_sphere/sie_dipole_impedance_matching.py \
    --config "${configuration}" \
    --run-dir "${run_directory}" \
    --executable "${executable}"

echo "  finished:      $(date --iso-8601=seconds)"
echo "  manifest:      ${run_directory}/run.json"
