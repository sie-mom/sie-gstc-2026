#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/mesh/mesh_utils.hpp"
#include "sie_mom/mesh/obj_mesh_loader.hpp"
#include "sie_mom/solvers/dielectric/dielectric_solver.hpp"
#include "sie_mom/solvers/surface_current_fields.hpp"
#include "sie_mom/utils/cli_helpers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace sie_mom;
using namespace sie_mom::solvers;

struct Arguments {
    fs::path mesh_path;
    fs::path output_directory;
    fs::path observations_path;
    fs::path output_fields_path;
    fs::path output_currents_path;

    real_t frequency_hz = real_t(0);
    real_t epsilon_r_interior = real_t(0);
    real_t mu_r_interior = real_t(0);
    real_t epsilon_r_exterior = real_t(0);
    real_t mu_r_exterior = real_t(0);

    int quadrature_rule = 17;
    real_t singularity_threshold_lambda0 = real_t(0.1);
    bool compute_residual = true;
    bool evaluate_fields = true;
    bool evaluate_2d_grid = true;

    DielectricFormulation formulation = DielectricFormulation::T_PMCHWT;
    NtlIntegralForm ntl_form = NtlIntegralForm::Surface;
    int line_quadrature_points = 20;

    int zx_grid_size = 80;
    real_t zx_span_radii = real_t(2);

    int radial_points = 100;
    real_t radial_theta_degrees = real_t(45);
    real_t radial_phi_degrees = real_t(0);
    real_t radial_min_radii = real_t(0.05);
    real_t radial_max_radii = real_t(2);

    sie_mom::excitation::SourceSpec source_spec = sie_mom::excitation::PlaneWaveExcitation{
        make_vec3(real_t(1), real_t(0), real_t(0)),
        make_vec3(real_t(0), real_t(0), real_t(1)),
        sie_mom::excitation::SourceRegion::exterior
    };
};

void print_usage(const char* program) {
    std::cout
        << "Usage:\n  " << program << " [options]\n\n"
        << "Required options:\n"
        << "  --mesh PATH                         Strict triangular OBJ mesh\n"
        << "  --output-dir PATH                   Result directory\n"
        << "  --frequency-hz VALUE                Frequency in hertz\n"
        << "  --epsilon-r-interior VALUE          Interior relative permittivity\n"
        << "  --mu-r-interior VALUE               Interior relative permeability\n"
        << "  --epsilon-r-exterior VALUE          Exterior relative permeability\n"
        << "  --mu-r-exterior VALUE               Exterior relative permeability\n"
        << "\nSource: unit x-polarized plane wave traveling +z in the exterior (region 2) by default.\n"
        << "Optional source controls:\n"
        << "  --effective-moment-interior VALUE   Centered z-directed electric dipole in interior\n"
        << "  --dipole REGION TYPE MX MY MZ PX PY PZ\n"
        << "                                      Dipole excitation: REGION (interior|exterior),\n"
        << "                                      TYPE (electric|magnetic), moment (MX MY MZ),\n"
        << "                                      position (PX PY PZ in meters)\n"
        << "  --plane-wave REGION EX EY EZ DX DY DZ\n"
        << "                                      Plane wave: REGION (interior|exterior),\n"
        << "                                      amplitude (EX EY EZ), direction (DX DY DZ)\n\n"
        << "Optional numerical controls:\n"
        << "  --formulation [t|n|muller]          Formulation: t for T-PMCHWT, n for N-PMCHWT, muller for N-Muller (default: t)\n"
        << "  --quadrature-rule VALUE             Dunavant rule (default: 17)\n"
        << "  --ntl-form [line|surface]           nTL integral: Form A line or Form B surface (default: surface)\n"
        << "  --line-quadrature-points VALUE      Gauss-Legendre points for Form A line (default: 20)\n"
        << "  --singularity-threshold-lambda0 V   Threshold/lambda0 (default: 0.1)\n"
        << "  --no-residual                       Skip residual reassembly\n"
        << "  --no-fields                         Skip all field evaluations\n\n"
        << "Optional field sampling:\n"
        << "  --observations PATH                 Custom observations CSV (point_index,x,y,z,region)\n"
        << "  --output-fields PATH                Custom output fields CSV path\n"
        << "  --output-currents PATH              Save PMCHWT physical currents to binary file\n"
        << "  --zx-grid-size VALUE                Points per ZX axis in y=0 plane (default: 80)\n"
        << "  --zx-span-radii VALUE               ZX half-span in radii (default: 2)\n"
        << "  --no-2d-grid                        Skip 2D ZX grid evaluation\n"
        << "  --radial-points VALUE               Radial samples (default: 100)\n"
        << "  --radial-theta-degrees VALUE        Radial-cut theta (default: 45)\n"
        << "  --radial-phi-degrees VALUE          Radial-cut phi (default: 0)\n"
        << "  --radial-min-radii VALUE            Minimum r/a (default: 0.05)\n"
        << "  --radial-max-radii VALUE            Maximum r/a (default: 2)\n"
        << "  -h, --help                          Show this message\n";
}

using namespace sie_mom::cli;

Arguments parse_arguments(int argc, char* argv[]) {
    Arguments arguments;

    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        if (option == "--mesh") {
            arguments.mesh_path = require_value(argc, argv, index);
        } else if (option == "--output-dir") {
            arguments.output_directory = require_value(argc, argv, index);
        } else if (option == "--frequency-hz") {
            arguments.frequency_hz = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--epsilon-r-interior") {
            arguments.epsilon_r_interior = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--mu-r-interior") {
            arguments.mu_r_interior = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--epsilon-r-exterior") {
            arguments.epsilon_r_exterior = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--mu-r-exterior") {
            arguments.mu_r_exterior = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--effective-moment-interior") {
            sie_mom::excitation::DipoleExcitation dipole;
            dipole.effective_moment = make_vec3(
                real_t(0), real_t(0), parse_real(require_value(argc, argv, index), option)
            );
            dipole.position = make_vec3(real_t(0), real_t(0), real_t(0));
            dipole.is_magnetic = false;
            dipole.region = sie_mom::excitation::SourceRegion::interior;
            arguments.source_spec = dipole;
        } else if (option == "--dipole") {
            const auto region = parse_region(require_value(argc, argv, index));
            const std::string type = require_value(argc, argv, index);
            if (type != "electric" && type != "magnetic") {
                throw std::invalid_argument("dipole type must be electric or magnetic");
            }
            sie_mom::excitation::DipoleExcitation dipole;
            dipole.region = region;
            dipole.is_magnetic = (type == "magnetic");
            dipole.effective_moment = parse_vector(argc, argv, index, option);
            dipole.position = parse_vector(argc, argv, index, option);
            arguments.source_spec = dipole;
        } else if (option == "--plane-wave") {
            const auto region = parse_region(require_value(argc, argv, index));
            sie_mom::excitation::PlaneWaveExcitation plane_wave;
            plane_wave.region = region;
            plane_wave.electric_amplitude = parse_vector(argc, argv, index, option);
            plane_wave.propagation_direction = parse_vector(argc, argv, index, option);
            arguments.source_spec = plane_wave;
        } else if (option == "--formulation") {
            const std::string val = require_value(argc, argv, index);
            if (val == "t" || val == "t_pmchwt") {
                arguments.formulation = DielectricFormulation::T_PMCHWT;
            } else if (val == "n" || val == "n_pmchwt") {
                arguments.formulation = DielectricFormulation::N_PMCHWT;
            } else if (val == "muller" || val == "n_muller") {
                arguments.formulation = DielectricFormulation::N_Muller;
            } else {
                throw std::invalid_argument("invalid formulation: " + val);
            }
        } else if (option == "--ntl-form") {
            const std::string val = require_value(argc, argv, index);
            if (val == "line") {
                arguments.ntl_form = NtlIntegralForm::Line;
            } else if (val == "surface") {
                arguments.ntl_form = NtlIntegralForm::Surface;
            } else {
                throw std::invalid_argument("invalid nTL form: " + val);
            }
        } else if (option == "--line-quadrature-points") {
            arguments.line_quadrature_points = parse_integer(require_value(argc, argv, index), option);
        } else if (option == "--quadrature-rule") {
            arguments.quadrature_rule = parse_integer(require_value(argc, argv, index), option);
        } else if (option == "--singularity-threshold-lambda0") {
            arguments.singularity_threshold_lambda0 = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--no-residual") {
            arguments.compute_residual = false;
        } else if (option == "--no-fields") {
            arguments.evaluate_fields = false;
        } else if (option == "--no-2d-grid") {
            arguments.evaluate_2d_grid = false;
        } else if (option == "--zx-grid-size") {
            arguments.zx_grid_size = parse_integer(require_value(argc, argv, index), option);
        } else if (option == "--zx-span-radii") {
            arguments.zx_span_radii = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--radial-points") {
            arguments.radial_points = parse_integer(require_value(argc, argv, index), option);
        } else if (option == "--radial-theta-degrees") {
            arguments.radial_theta_degrees = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--radial-phi-degrees") {
            arguments.radial_phi_degrees = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--radial-min-radii") {
            arguments.radial_min_radii = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--radial-max-radii") {
            arguments.radial_max_radii = parse_real(require_value(argc, argv, index), option);
        } else if (option == "--observations") {
            arguments.observations_path = require_value(argc, argv, index);
        } else if (option == "--output-fields") {
            arguments.output_fields_path = require_value(argc, argv, index);
        } else if (option == "--output-currents") {
            arguments.output_currents_path = require_value(argc, argv, index);
        } else if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    if (arguments.output_directory.empty() && !arguments.output_fields_path.empty()) {
        arguments.output_directory = arguments.output_fields_path.parent_path();
    }
    if (arguments.output_directory.empty() && !arguments.output_currents_path.empty()) {
        arguments.output_directory = arguments.output_currents_path.parent_path();
    }
    if (arguments.mesh_path.empty() || arguments.output_directory.empty()) {
        throw std::invalid_argument("mesh and output-dir (or output-fields) are required");
    }
    return arguments;
}

struct ObservationSamples {
    std::vector<vec3_t> points;
    std::vector<int> regions;
    std::size_t zx_count = 0;
    std::size_t radial_count = 0;
};

ObservationSamples load_observation_samples(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open observations CSV: " + path.string());
    }
    std::string line;
    if (!std::getline(stream, line)) {
        throw std::runtime_error("empty observations CSV: " + path.string());
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();

    std::stringstream header_stream(line);
    std::string col;
    std::vector<std::string> cols;
    while (std::getline(header_stream, col, ',')) cols.push_back(col);
    int x_col = -1, y_col = -1, z_col = -1, reg_col = -1;
    for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
        if (cols[i] == "x") x_col = i;
        else if (cols[i] == "y") y_col = i;
        else if (cols[i] == "z") z_col = i;
        else if (cols[i] == "region") reg_col = i;
    }
    if (x_col < 0 || y_col < 0 || z_col < 0 || reg_col < 0) {
        throw std::runtime_error("observations CSV must contain x, y, z, region columns: " + path.string());
    }

    ObservationSamples obs;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<std::string> cells;
        while (std::getline(ss, col, ',')) cells.push_back(col);
        if (cells.size() < cols.size()) continue;
        const real_t x = static_cast<real_t>(std::stod(cells[x_col]));
        const real_t y = static_cast<real_t>(std::stod(cells[y_col]));
        const real_t z = static_cast<real_t>(std::stod(cells[z_col]));
        const int reg = std::stoi(cells[reg_col]);
        obs.points.push_back(make_vec3(x, y, z));
        obs.regions.push_back(reg);
    }
    obs.zx_count = obs.points.size();
    obs.radial_count = 0;
    return obs;
}

ObservationSamples make_observation_samples(const Arguments& args, real_t radius) {
    ObservationSamples observations;

    if (args.evaluate_2d_grid && args.zx_grid_size > 0) {
        observations.zx_count = static_cast<std::size_t>(args.zx_grid_size)
            * static_cast<std::size_t>(args.zx_grid_size);
        const real_t span = args.zx_span_radii * radius;
        const real_t zx_denom = (args.zx_grid_size > 1)
            ? static_cast<real_t>(args.zx_grid_size - 1) : real_t(1);
        for (int z_idx = 0; z_idx < args.zx_grid_size; ++z_idx) {
            const real_t z = -span + real_t(2) * span * static_cast<real_t>(z_idx) / zx_denom;
            for (int x_idx = 0; x_idx < args.zx_grid_size; ++x_idx) {
                const real_t x = -span + real_t(2) * span * static_cast<real_t>(x_idx) / zx_denom;
                const vec3_t pt = make_vec3(x, real_t(0), z);
                observations.points.push_back(pt);
                const real_t r = std::sqrt(x * x + z * z);
                observations.regions.push_back(r < radius ? 1 : 2);
            }
        }
    }

    if (args.radial_points > 0) {
        observations.radial_count = static_cast<std::size_t>(args.radial_points);
        const real_t pi = std::acos(real_t(-1));
        const real_t theta = args.radial_theta_degrees * pi / real_t(180);
        const real_t phi = args.radial_phi_degrees * pi / real_t(180);

        const vec3_t direction = {
            std::sin(theta) * std::cos(phi),
            std::sin(theta) * std::sin(phi),
            std::cos(theta),
        };

        const real_t span = args.radial_max_radii - args.radial_min_radii;
        const real_t denom = (args.radial_points > 1)
            ? static_cast<real_t>(args.radial_points - 1) : real_t(1);
        for (int step = 0; step < args.radial_points; ++step) {
            const real_t fraction = static_cast<real_t>(step) / denom;
            const real_t r = radius * (args.radial_min_radii + span * fraction);
            observations.points.push_back({
                r * direction.x,
                r * direction.y,
                r * direction.z,
            });
            observations.regions.push_back(r < radius ? 1 : 2);
        }
    }
    return observations;
}

using FieldSample = sie_mom::ElectromagneticFieldSample;
using sie_mom::subtract_fields;

std::vector<FieldSample> evaluate_all_fields(
    const sie_mom::mesh::MeshData& mesh,
    const std::vector<std::complex<real_t>>& j,
    const std::vector<std::complex<real_t>>& m,
    const ObservationSamples& observations,
    const DielectricOptions& options,
    const sie_mom::excitation::PreparedExcitation& source
) {

    const auto exterior_surface_integrals = sie_mom::solvers::evaluate_surface_integral_fields(
        mesh, j, m, observations.points,
        options.frequency_hz, options.exterior, options.quadrature_rule,
        options.vacuum_permittivity, options.vacuum_permeability
    );
    

    auto interior_j = j;
    auto interior_m = m;
    // The interior currents are the negative of the exterior currents.
    for (auto& value : interior_j) value = -value;
    for (auto& value : interior_m) value = -value;
    const auto interior_surface_integrals = sie_mom::solvers::evaluate_surface_integral_fields(
        mesh, interior_j, interior_m, observations.points,
        options.frequency_hz, options.interior, options.quadrature_rule,
        options.vacuum_permittivity, options.vacuum_permeability
    );
    
    const auto incident_fields = sie_mom::excitation::evaluate_incident_fields(
        observations.points, source
    );

    std::vector<FieldSample> physical;
    physical.reserve(observations.points.size());
    const FieldSample zero_incident_field{};
    for (std::size_t index = 0; index < observations.points.size(); ++index) {
        const bool source_is_interior =
            source.region() == sie_mom::excitation::SourceRegion::interior;
        auto e_int = subtract_fields(
            source_is_interior ? incident_fields[index] : zero_incident_field,
            interior_surface_integrals[index]
        );
        auto e_ext = subtract_fields(
            source_is_interior ? zero_incident_field : incident_fields[index],
            exterior_surface_integrals[index]
        );
        physical.push_back(observations.regions[index] == 1 ? e_int : e_ext);
    }
    return physical;
}

void write_field_csv(
    const fs::path& path,
    const ObservationSamples& observations,
    const std::vector<FieldSample>& fields,
    std::size_t offset = 0,
    std::size_t count = 0,
    bool standard_header = false
) {
    if (count == 0) {
        count = (offset < observations.points.size()) ? (observations.points.size() - offset) : 0;
    }
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open output CSV: " + path.string());
    }
    file << std::scientific << std::setprecision(12);
    if (standard_header) {
        file << "point_index,x,y,z,region,"
             << "Ex_real,Ex_imag,Ey_real,Ey_imag,Ez_real,Ez_imag,"
             << "Hx_real,Hx_imag,Hy_real,Hy_imag,Hz_real,Hz_imag\n";
    } else {
        file << "x,y,z,region,"
             << "E_x_re,E_x_im,E_y_re,E_y_im,E_z_re,E_z_im,"
             << "H_x_re,H_x_im,H_y_re,H_y_im,H_z_re,H_z_im\n";
    }

    for (std::size_t i = offset; i < offset + count; ++i) {
        const auto& pt = observations.points[i];
        const auto& f = fields[i];
        if (standard_header) {
            file << (i - offset) << ",";
        }
        file << pt.x << "," << pt.y << "," << pt.z << "," << observations.regions[i] << ","
             << f.electric[0].real() << "," << f.electric[0].imag() << ","
             << f.electric[1].real() << "," << f.electric[1].imag() << ","
             << f.electric[2].real() << "," << f.electric[2].imag() << ","
             << f.magnetic[0].real() << "," << f.magnetic[0].imag() << ","
             << f.magnetic[1].real() << "," << f.magnetic[1].imag() << ","
             << f.magnetic[2].real() << "," << f.magnetic[2].imag() << "\n";
    }
}

void print_summary(const sie_mom::solvers::DielectricResult& result, const char* formulation_name) {
    const auto& metrics = result.metrics;
    std::cout << std::scientific << std::setprecision(6)
        << formulation_name << " summary:\n"
        << "  vertices: " << metrics.vertex_count << "\n"
        << "  faces: " << metrics.face_count << "\n"
        << "  edges: " << metrics.edge_count << "\n"
        << "  matrix_dimension: " << metrics.matrix_dimension << "\n"
        << "  dense_matrix_gib: "
        << static_cast<double>(metrics.dense_matrix_bytes) / (1024.0 * 1024.0 * 1024.0) << "\n"
        << "  device_preparation_seconds: " << metrics.device_preparation_seconds << "\n"
        << "  assembly_seconds: " << metrics.assembly_seconds << "\n"
        << "  solve_seconds: " << metrics.solve_seconds << "\n";
    if (metrics.residual_computed) {
        std::cout
            << "  residual_seconds: " << metrics.residual_seconds << "\n"
            << "  scaled_relative_residual: " << metrics.relative_residual << "\n";
    }
    std::cout
        << "  max_abs_J: " << metrics.maximum_j_coefficient << "\n"
        << "  max_abs_M: " << metrics.maximum_m_coefficient << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);

        std::cout << "--- PMCHWT Dielectric Sphere Solver ---\n";
        std::cout << "Loading mesh from " << arguments.mesh_path << "...\n";
        sie_mom::mesh::ObjMeshLoader loader(arguments.mesh_path.string());
        const auto& mesh = loader.getMeshData();

        const std::size_t matrix_dimension = 2 * mesh.edges.size();
        const long double dense_matrix_gib =
            static_cast<long double>(matrix_dimension)
            * static_cast<long double>(matrix_dimension)
            * static_cast<long double>(sizeof(complex__float))
            / (1024.0L * 1024.0L * 1024.0L);

        std::cout << "Loaded mesh:\n"
            << "  vertices: " << mesh.vertices.size() << "\n"
            << "  faces: " << mesh.faces.size() << "\n"
            << "  edges: " << mesh.edges.size() << "\n"
            << "  matrix_dimension: " << matrix_dimension << "\n"
            << "  dense_matrix_gib: " << std::fixed << std::setprecision(3)
            << static_cast<double>(dense_matrix_gib) << "\n";

        DielectricOptions options;
        options.frequency_hz = arguments.frequency_hz;
        options.interior.relative_permittivity = arguments.epsilon_r_interior;
        options.interior.relative_permeability = arguments.mu_r_interior;
        options.exterior.relative_permittivity = arguments.epsilon_r_exterior;
        options.exterior.relative_permeability = arguments.mu_r_exterior;
        options.quadrature_rule = arguments.quadrature_rule;
        options.singularity_threshold_lambda0 = arguments.singularity_threshold_lambda0;
        options.compute_residual = arguments.compute_residual;
        options.formulation = arguments.formulation;
        options.ntl_form = arguments.ntl_form;
        options.line_quadrature_points = arguments.line_quadrature_points;

        const auto source = sie_mom::excitation::prepare_source(
            arguments.source_spec, options.frequency_hz, options.interior, options.exterior,
            options.vacuum_permittivity, options.vacuum_permeability
        );
        std::visit([](const auto& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, sie_mom::excitation::PlaneWaveExcitation>) {
                std::cout << "Excitation: "
                          << (s.region == sie_mom::excitation::SourceRegion::interior ? "interior" : "exterior")
                          << " plane wave\n";
            } else if constexpr (std::is_same_v<T, sie_mom::excitation::DipoleExcitation>) {
                std::cout << "Excitation: "
                          << (s.region == sie_mom::excitation::SourceRegion::interior ? "interior" : "exterior")
                          << " " << (s.is_magnetic ? "magnetic" : "electric") << " dipole\n";
            }
        }, arguments.source_spec);

        const char* form_name = "T-PMCHWT";
        if (arguments.formulation == DielectricFormulation::N_PMCHWT) {
            form_name = "N-PMCHWT";
        } else if (arguments.formulation == DielectricFormulation::N_Muller) {
            form_name = "N-Muller";
        }
        if (arguments.formulation == DielectricFormulation::N_PMCHWT ||
            arguments.formulation == DielectricFormulation::N_Muller) {
            std::cout << form_name << " nTL integral: "
                      << (arguments.ntl_form == NtlIntegralForm::Line
                          ? "line (Form A)" : "surface (Form B)");
            if (arguments.ntl_form == NtlIntegralForm::Line) {
                std::cout << ", Gauss-Legendre points: " << arguments.line_quadrature_points;
            }
            std::cout << "\n";
        }
        std::cout << "Starting dense CUDA solve (" << form_name << ")...\n" << std::flush;
        const auto result = solve_dielectric_dense(mesh, options, source);

        print_summary(result, form_name);

        if (!arguments.output_currents_path.empty()) {
            if (!arguments.output_currents_path.parent_path().empty()) {
                fs::create_directories(arguments.output_currents_path.parent_path());
            }
            const std::size_t e = mesh.edges.size();
            std::vector<double> raw(8 * e);
            for (std::size_t i = 0; i < e; ++i) {
                // J1 = -J
                raw[2 * i] = -static_cast<double>(result.physical_currents[i].real());
                raw[2 * i + 1] = -static_cast<double>(result.physical_currents[i].imag());
                // M1 = -M
                raw[2 * (e + i)] = -static_cast<double>(result.physical_currents[e + i].real());
                raw[2 * (e + i) + 1] = -static_cast<double>(result.physical_currents[e + i].imag());
                // J2 = J
                raw[2 * (2 * e + i)] = static_cast<double>(result.physical_currents[i].real());
                raw[2 * (2 * e + i) + 1] = static_cast<double>(result.physical_currents[i].imag());
                // M2 = M
                raw[2 * (3 * e + i)] = static_cast<double>(result.physical_currents[e + i].real());
                raw[2 * (3 * e + i) + 1] = static_cast<double>(result.physical_currents[e + i].imag());
            }
            std::ofstream stream(arguments.output_currents_path, std::ios::binary);
            if (!stream) {
                throw std::runtime_error("failed to open output currents file: " + arguments.output_currents_path.string());
            }
            stream.write(reinterpret_cast<const char*>(raw.data()),
                         static_cast<std::streamsize>(raw.size() * sizeof(double)));
            std::cout << "PMCHWT physical currents saved to " << arguments.output_currents_path << "\n";
        }

        if (arguments.evaluate_fields) {
            fs::create_directories(arguments.output_directory);
            ObservationSamples observations;
            const bool has_custom_obs = !arguments.observations_path.empty();
            if (has_custom_obs) {
                observations = load_observation_samples(arguments.observations_path);
            } else {
                const real_t radius = sie_mom::mesh::infer_sphere_radius(mesh);
                observations = make_observation_samples(arguments, radius);
            }

            std::cout << "Evaluating near fields (" << observations.points.size() << " total points";
            if (!has_custom_obs) {
                std::cout << ": " << observations.zx_count << " 2D grid, "
                          << observations.radial_count << " radial";
            }
            std::cout << ")...\n";

            std::vector<std::complex<real_t>> j(mesh.edges.size());
            std::vector<std::complex<real_t>> m(mesh.edges.size());
            for(std::size_t i = 0; i < mesh.edges.size(); ++i) {
                j[i] = result.physical_currents[i];
                m[i] = result.physical_currents[mesh.edges.size() + i];
            }

            const auto fields = evaluate_all_fields(mesh, j, m, observations, options, source);

            if (has_custom_obs) {
                const auto out_path = !arguments.output_fields_path.empty()
                    ? arguments.output_fields_path
                    : (arguments.output_directory / "fields.csv");
                if (!out_path.parent_path().empty()) {
                    fs::create_directories(out_path.parent_path());
                }
                write_field_csv(out_path, observations, fields, 0, observations.points.size(), true);
                std::cout << "  Observation fields written to " << out_path << "\n";
            } else {
                if (observations.zx_count > 0) {
                    const auto zx_path = arguments.output_directory / "zx_fields.csv";
                    write_field_csv(zx_path, observations, fields, 0, observations.zx_count, false);
                    std::cout << "  2D ZX grid fields written to " << zx_path << "\n";
                }
                if (observations.radial_count > 0) {
                    const auto radial_path = arguments.output_directory / "radial_fields.csv";
                    write_field_csv(radial_path, observations, fields, observations.zx_count, observations.radial_count, false);
                    std::cout << "  Radial fields written to " << radial_path << "\n";
                }
            }
        }

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
