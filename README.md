# SIE-GSTC publication code

This repository accompanies *SIE-GSTC Formulation for Closed Bianisotropic
Metasurfaces with Passive Lossless Synthesis*, by German Farinas Perez and
Jordan Budhu (submitted).

Software developer: **German Farinas Perez**. Contact: [germanf@vt.edu](mailto:germanf@vt.edu).

The C++/CUDA solvers and Python workflows reproduce three numerical examples:

- [Example 1: omega-biisotropic impedance matching](applications/omega_biisotropic_sphere/publication/README.md).
- [Example 2: anisotropic polarization rotation, Parts A and B](applications/anisotropic_gstc_sphere/publication/README.md).
- [Example 3: a GSTC metasurface on a circular-arc spindle](applications/gstc_surface/publication/README.md).

Start with the [quick start](docs/source/quickstart.rst) for software and GPU
requirements, installation, compilation, and reproduction commands.

```bash
git clone https://github.com/sie-mom/sie-gstc-2026.git
cd sie-gstc-2026
```

## Documentation site (GitHub Pages)

The Sphinx + Doxygen docs in `docs/` are built by `.github/workflows/deploy-docs.yml` on pushes and pull requests targeting `main`. Publishing to GitHub Pages is gated behind the repository variable `DOCS_PUBLISH`:

- **Settings -> Secrets and variables -> Actions -> Variables -> `DOCS_PUBLISH`**
- `true` -> after GitHub Pages is configured to use GitHub Actions, the `deploy` job publishes or updates `https://sie-mom.github.io/sie-gstc-2026/`
- `false` (or unset) -> only the `build` job runs; the site is not touched

Flip this variable when you're ready to make the docs public (e.g. after the paper is published).

## C++ project compilation

The C++ part of the project uses CMake. The standard workflow is to configure, build, and then test.

### 1. Configure
From the project root directory, run CMake to generate the build system. This only needs to be done once, or after changing `CMakeLists.txt` files.

Important: `SIE_MOM_USE_DOUBLE_PRECISION` is a CMake cache variable. The `ON` default in `CMakeLists.txt` only applies the first time a build directory is configured. After that, the value stored in the build directory cache wins. To avoid ambiguity, always pass the precision explicitly when configuring.

Choose one of the following build types:

```bash
# For development and debugging (includes debug symbols, no optimizations)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSIE_MOM_USE_DOUBLE_PRECISION=ON

# For performance and deployment (optimized, no debug symbols)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSIE_MOM_USE_DOUBLE_PRECISION=ON
```

If you intentionally want single precision, configure it explicitly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSIE_MOM_USE_DOUBLE_PRECISION=OFF
```

If you are building CUDA code, set the CUDA architecture explicitly to match your GPU.
This avoids architecture-dependent compile issues across different machines.

```bash
# Example: A100 (compute capability 8.0)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSIE_MOM_USE_DOUBLE_PRECISION=ON -DCMAKE_CUDA_ARCHITECTURES=80

# Example: RTX 3060 (compute capability 8.6)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSIE_MOM_USE_DOUBLE_PRECISION=ON -DCMAKE_CUDA_ARCHITECTURES=86
```

Useful values:
- 80: A100
- 86: RTX 3060 / RTX 30xx class GPUs
- 89: RTX 40xx
- 90: H100

To check your GPU compute capability:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

If you change architecture flags or precision, do a clean reconfigure:

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSIE_MOM_USE_DOUBLE_PRECISION=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

#### Windows: PowerShell, Ninja, and CUDA

The Windows build was verified with Visual Studio 2022 Build Tools (MSVC 19.44),
CUDA 12.9, and Ninja in the `comsol` Conda environment. That CUDA version rejects
the MSVC 19.51 compiler selected by Visual Studio 2026. Ninja also avoids requiring
CUDA's Visual Studio build integration.

From the project root, activate the environment containing CMake, Ninja, and the
CUDA Toolkit, then load the VS 2022 compiler environment in the same terminal.
Adjust the Visual Studio script path if your installation is elsewhere.

```powershell
conda activate comsol

& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

$cxx = (Get-Command cl.exe -ErrorAction Stop).Source

cmake --fresh -S . -B build -G Ninja `
  "-DCMAKE_BUILD_TYPE=Release" `
  "-DCMAKE_CXX_COMPILER=$cxx" `
  "-DCMAKE_CUDA_HOST_COMPILER=$cxx" `
  "-DCMAKE_CUDA_COMPILER=$env:CONDA_PREFIX/Library/bin/nvcc.exe" `
  "-DCUDAToolkit_ROOT=$env:CONDA_PREFIX/Library" `
  "-DSIE_MOM_USE_DOUBLE_PRECISION=ON" `
  "-DCMAKE_CUDA_ARCHITECTURES=80"
```

Quote each complete `-D` argument in PowerShell. The `--fresh` option requires
CMake 3.24 or later and resets the configuration when switching generators;
omit it for routine reconfiguration. Keep `build` as the output directory for
application scripts that use that predefined path. Change architecture `80`
to match the target GPU as described above.

After configuration succeeds, build with four jobs and run the default C++/CUDA
test suite:

```powershell
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

To save the build output while displaying it in the terminal:

```powershell
cmake --build build --parallel 4 2>&1 |
    Tee-Object -FilePath build/compile.log
```

To select all Catch2 cases explicitly, including hidden cases:

```powershell
.\build\tests\cpp\run_cpp_tests.exe "*" --durations yes
```

The hidden PMCHWT test is marked `!mayfail`; inspect the test summary as well as
the exit code. The verified Windows Release/double-precision build passed CTest,
and the user-run Catch2 suite reported 37,121 assertions across 88 test cases.
**Linux verification is pending:** rebuild and rerun the C++/CUDA tests on Linux
before treating these portability changes as validated there. Use a separate
build directory when switching operating systems.

### 2. Build
Compile the code using the build system generated by CMake.

```bash
cmake --build build
```

## Running C++ Tests with Catch2

Catch2 test filtering works by matching `TEST_CASE` names or tags, not `SECTION` names. For example, to run all tests with a specific tag or name:

```bash
./build/tests/cpp/run_cpp_tests "[TLMatrixAssembly][SphericalSurface]"
```
or
```bash
./build/tests/cpp/run_cpp_tests "Assembly TL operator on spherical surface mesh"
```

**Note:** You cannot run a specific `SECTION` by name. All `SECTION`s inside a matching `TEST_CASE` will be run in sequence. If you want to run only a specific part (like the solver), move it to its own `TEST_CASE` or comment out other sections.


## Installation (local editable)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### Mesh (Gmsh) Support
For mesh generation using Gmsh:
```bash
pip install -e .[mesh]
```

### Optional NGSolve validation

NGSolve can use the same virtual environment as the SIE workflows. Installing the `[ngsolve]` extra includes NGSolve along with Intel MKL (`mkl`, `intel-openmp`), which enables NGSolve's parallel PARDISO sparse direct solver across all available CPU threads:

```bash
pip install -e ".[ngsolve]"
```

> [!TIP]
> Intel MKL enables NGSolve's multithreaded PARDISO sparse direct solver (`--inverse auto` or `--inverse pardiso`). Without MKL, NGSolve falls back to single-threaded UMFPACK, which is significantly slower for large 3D FEM problems with hundreds of thousands of DOFs.

The reusable Maxwell/GSTC backend lives in
[`src/sie_mom/validation/ngsolve/`](src/sie_mom/validation/ngsolve/).
Application runners provide the [sphere validation workflow](applications/anisotropic_gstc_sphere/README.md#optional-ngsolve-validation)
and [general-surface validation workflow](applications/gstc_surface/README.md#ngsolve-with-the-same-sie-interface).
The latter preserves the SIE surface triangles, physical tensors and local
frames. Its measured ogive comparisons and current accuracy limits are recorded
in the application README. NGSolve is optional for all SIE-only workflows.

### GPU (CuPy) Support
Select one extra:
- `cuda13` → `cupy-cuda13x` (local machine with CUDA 13 driver/toolkit)
- `cuda12` → `cupy-cuda12x` (HPC modules like cuda12.3 / CUDA/12.6.0)

Examples:
```bash
# Local (CUDA 13):
pip install -e .[cuda13]

# HPC (CUDA 12.x module):
pip install -e .[cuda12]

# Development with CUDA 13 + tooling:
pip install -e .[cuda13,dev]
```

Tip: On HPC, load a CUDA 12.x module BEFORE installing.

## Quick Test
```bash
python - <<'PY'
from sie_mom.mesh import SphericalSurfaceMesh
m = SphericalSurfaceMesh(refinement_level=2).generate()
print(m.summary())
PY
```

## Development Tools
- Ruff (lint/import sort): `ruff check .`
- Mypy (basic type check): `mypy src`
- Pytest: (tests directory; minimal tests added later)
- Tox (multi-Python): `tox -p` (will skip missing interpreters)

## Documentation Tools
- Sphinx (builder): `pip install -e .[docs]`
- Sphinx Book Theme (HTML theme, via docs extra)
- MyST-Parser (Markdown support)
- Doxygen (CUDA/C++ reference): `sudo apt install doxygen`
- Build HTML docs: `make -C docs html`
- Build PDF docs: `make -C docs pdf`

`make -C docs pdf` is the repository's PDF target; it runs Sphinx's
`latexpdf` builder after Doxygen. The resulting PDF is
`docs/build/latex/sie_mom.pdf`. Doxygen may also create a separate
`docs/latex/build/refman.pdf`; it is not used by the Sphinx documentation
site or by the main PDF build.

### Previewing Documentation
1. After running `make html`, open `docs/build/html/index.html` in a browser (e.g. `xdg-open` on Linux or `open` on macOS).
2. Alternatively, serve the directory with `python -m http.server --directory docs/_build/html` and visit `http://localhost:8000`.

## HPC Usage Outline
```bash
# On HPC interactive node
git clone <your repo>
module load Python/3.11.5-GCCcore-13.2.0
# Load a CUDA module first, matching chosen wheel or source build
# module load cuda12.3/toolkit/12.3.2   # or CUDA/12.6.0 etc.
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .[cuda13]        # .[cuda13]
pip install -e .[cuda13,dev]    # include tests & linters
```

# Github key setup

## Git Authentication Setup (Recommended)

To avoid entering your username and password/token every time you interact with GitHub, it is highly recommended to use SSH keys. This is a one-time setup per machine that is both more secure and much more convenient.

### 1. Generate a New SSH Key

Run the following command on the machine you want to authenticate from (e.g., the HPC login node). Replace the email with the one you use for GitHub.

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

When prompted, press Enter to accept the default file location and Enter twice more to create the key without a passphrase for maximum convenience.

### 2. Add the SSH Key to Your GitHub Account

First, display your new public key in the terminal and copy its entire output.

```bash
cat ~/.ssh/id_ed25519.pub
```

Next, go to your GitHub account in a web browser:
1. Go to Settings (click your profile picture in the top-right).
2. In the left sidebar, click SSH and GPG keys.
3. Click New SSH key.
4. Give it a descriptive Title (e.g., "HPC Falcon Server").
5. Paste the key you copied into the Key text box.
6. Click Add SSH key.

### 3. Configure Your Local Repository to Use SSH

Finally, tell your local repository clone to use the SSH protocol instead of HTTPS.

```bash
git remote set-url origin git@github.com:sie-mom/sie-gstc-2026.git
```

After this, all git pull and git push commands will work automatically without asking for credentials.

## License
MIT
