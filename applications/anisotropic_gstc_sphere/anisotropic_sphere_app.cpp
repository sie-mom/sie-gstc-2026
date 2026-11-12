#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/gstc_utils.hpp"
#include "sie_mom/mesh/mesh_utils.hpp"
#include "sie_mom/mesh/obj_mesh_loader.hpp"
#include "sie_mom/solvers/gstc_sie.hpp"
#include "sie_mom/solvers/surface_current_fields.hpp"
#include "sie_mom/utils/cli_helpers.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct ExcitationCase {
    std::string id;
    std::optional<sie_mom::excitation::SourceSpec> interior;
    std::optional<sie_mom::excitation::SourceSpec> exterior;
};

struct Arguments {
    fs::path mesh_path;
    fs::path tensor_path;
    fs::path output_directory;

    real_t frequency_hz = real_t(0);
    real_t epsilon_r_interior = real_t(0);
    real_t mu_r_interior = real_t(0);
    real_t epsilon_r_exterior = real_t(0);
    real_t mu_r_exterior = real_t(0);
    std::vector<ExcitationCase> cases;

    int quadrature_rule = 17;
    real_t singularity_threshold_lambda0 = real_t(0.1);
    sie_mom::GstcBasisType basis = sie_mom::GstcBasisType::spherical;
    bool compute_residual = true;
    bool evaluate_fields = true;

    int zx_grid_size = 200;
    real_t zx_span_radii = real_t(2);
    int radial_points = 300;
    real_t radial_theta_degrees = real_t(45);
    real_t radial_phi_degrees = real_t(0);
    real_t radial_min_radii = real_t(0.05);
    real_t radial_max_radii = real_t(2);
};

const char* basis_name(sie_mom::GstcBasisType basis) {
    return basis == sie_mom::GstcBasisType::face ? "face" : "spherical";
}

sie_mom::GstcBasisType parse_basis(const std::string& value) {
    if (value == "spherical") {
        return sie_mom::GstcBasisType::spherical;
    }
    if (value == "face") {
        return sie_mom::GstcBasisType::face;
    }
    throw std::invalid_argument("--basis must be 'spherical' or 'face', got: " + value);
}

void print_usage(const char* program) {
    std::cout
        << "Usage:\n  " << program << " [options]\n\n"
        << "Required options:\n"
        << "  --mesh PATH                         Strict triangular OBJ mesh\n"
        << "  --tensors PATH                      Per-face GSTC tensor binary\n"
        << "  --output-dir PATH                   Shared run directory\n"
        << "  --frequency-hz VALUE                Frequency in hertz\n"
        << "  --epsilon-r-interior VALUE          Interior relative permittivity\n"
        << "  --mu-r-interior VALUE               Interior relative permeability\n"
        << "  --epsilon-r-exterior VALUE          Exterior relative permittivity\n"
        << "  --mu-r-exterior VALUE               Exterior relative permeability\n"
        << "\nSimulation cases (repeat --case; one source per region):\n"
        << "  --case ID                           Start an independent RHS case\n"
        << "  --dipole REGION TYPE PX PY PZ X Y Z  Real effective moment and position (m)\n"
        << "  --plane-wave REGION EX EY EZ DX DY DZ  Real E0 and propagation direction\n"
        << "  REGION: interior or exterior; TYPE: electric or magnetic\n\n"
        << "Optional numerical controls:\n"
        << "  --basis {spherical,face}            Local tangent basis for GSTC tensors (default: spherical)\n"
        << "  --quadrature-rule VALUE             Dunavant rule (default: 17)\n"
        << "  --singularity-threshold-lambda0 V   Threshold/lambda0 (default: 0.1)\n"
        << "  --no-residual                       Skip residual reassembly\n"
        << "  --no-fields                         Skip 2D and radial field evaluation\n\n"
        << "Optional field sampling:\n"
        << "  --zx-grid-size VALUE                Even points per ZX axis (default: 200)\n"
        << "  --zx-span-radii VALUE               ZX half-span/a (default: 2)\n"
        << "  --radial-points VALUE               Radial samples (default: 300)\n"
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
        } else if (option == "--tensors") {
            arguments.tensor_path = require_value(argc, argv, index);
        } else if (option == "--output-dir") {
            arguments.output_directory = require_value(argc, argv, index);
        } else if (option == "--frequency-hz") {
            arguments.frequency_hz = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--epsilon-r-interior") {
            arguments.epsilon_r_interior = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--mu-r-interior") {
            arguments.mu_r_interior = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--epsilon-r-exterior") {
            arguments.epsilon_r_exterior = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--mu-r-exterior") {
            arguments.mu_r_exterior = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--case") {
            arguments.cases.push_back({require_value(argc, argv, index), {}, {}});
        } else if (option == "--dipole" || option == "--plane-wave") {
            if (arguments.cases.empty()) {
                throw std::invalid_argument(option + " requires a preceding --case");
            }
            const auto region = parse_region(require_value(argc, argv, index));
            auto& slot = region == sie_mom::excitation::SourceRegion::interior
                ? arguments.cases.back().interior : arguments.cases.back().exterior;
            if (slot) {
                throw std::invalid_argument("duplicate source region in case " + arguments.cases.back().id);
            }
            if (option == "--dipole") {
                const std::string type = require_value(argc, argv, index);
                if (type != "electric" && type != "magnetic") {
                    throw std::invalid_argument("dipole type must be electric or magnetic");
                }
                sie_mom::excitation::DipoleExcitation dipole;
                dipole.region = region;
                dipole.is_magnetic = type == "magnetic";
                dipole.effective_moment = parse_vector(argc, argv, index, option);
                dipole.position = parse_vector(argc, argv, index, option);
                slot = dipole;
            } else {
                sie_mom::excitation::PlaneWaveExcitation plane_wave;
                plane_wave.region = region;
                plane_wave.electric_amplitude = parse_vector(argc, argv, index, option);
                plane_wave.propagation_direction = parse_vector(argc, argv, index, option);
                slot = plane_wave;
            }
        } else if (option == "--basis") {
            arguments.basis = parse_basis(require_value(argc, argv, index));
        } else if (option == "--quadrature-rule") {
            arguments.quadrature_rule = parse_integer(
                require_value(argc, argv, index), option
            );
        } else if (option == "--singularity-threshold-lambda0") {
            arguments.singularity_threshold_lambda0 = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--no-residual") {
            arguments.compute_residual = false;
        } else if (option == "--no-fields") {
            arguments.evaluate_fields = false;
        } else if (option == "--zx-grid-size") {
            arguments.zx_grid_size = parse_integer(
                require_value(argc, argv, index), option
            );
        } else if (option == "--zx-span-radii") {
            arguments.zx_span_radii = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--radial-points") {
            arguments.radial_points = parse_integer(
                require_value(argc, argv, index), option
            );
        } else if (option == "--radial-theta-degrees") {
            arguments.radial_theta_degrees = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--radial-phi-degrees") {
            arguments.radial_phi_degrees = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--radial-min-radii") {
            arguments.radial_min_radii = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--radial-max-radii") {
            arguments.radial_max_radii = parse_real(
                require_value(argc, argv, index), option
            );
        } else if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    if (arguments.mesh_path.empty()) {
        throw std::invalid_argument("--mesh is required");
    }
    if (arguments.tensor_path.empty()) {
        throw std::invalid_argument("--tensors is required");
    }
    if (arguments.output_directory.empty()) {
        throw std::invalid_argument("--output-dir is required");
    }
    if (!(arguments.frequency_hz > real_t(0))) {
        throw std::invalid_argument("--frequency-hz must be positive");
    }
    if (!(arguments.epsilon_r_interior > real_t(0))
        || !(arguments.mu_r_interior > real_t(0))
        || !(arguments.epsilon_r_exterior > real_t(0))
        || !(arguments.mu_r_exterior > real_t(0))) {
        throw std::invalid_argument("all relative material parameters must be positive");
    }
    if (arguments.cases.empty()) {
        throw std::invalid_argument("at least one --case is required");
    }
    std::set<std::string> case_ids;
    const std::regex identifier("[a-z0-9][a-z0-9_-]*");
    for (const auto& simulation_case : arguments.cases) {
        if (!std::regex_match(simulation_case.id, identifier)) {
            throw std::invalid_argument("invalid case ID: " + simulation_case.id);
        }
        if (!case_ids.insert(simulation_case.id).second) {
            throw std::invalid_argument("duplicate case ID: " + simulation_case.id);
        }
        if (!simulation_case.interior && !simulation_case.exterior) {
            throw std::invalid_argument("case " + simulation_case.id + " requires an excitation");
        }
    }
    if (arguments.quadrature_rule < 1 || arguments.quadrature_rule > 20) {
        throw std::invalid_argument("--quadrature-rule must be in [1, 20]");
    }
    if (!(arguments.singularity_threshold_lambda0 > real_t(0))) {
        throw std::invalid_argument(
            "--singularity-threshold-lambda0 must be positive"
        );
    }
    if (arguments.zx_grid_size < 2 || arguments.zx_grid_size % 2 != 0) {
        throw std::invalid_argument(
            "--zx-grid-size must be an even integer greater than one"
        );
    }
    if (!(arguments.zx_span_radii > real_t(1))) {
        throw std::invalid_argument("--zx-span-radii must be greater than one");
    }
    if (arguments.radial_points < 2) {
        throw std::invalid_argument("--radial-points must be at least two");
    }
    if (!(arguments.radial_min_radii > real_t(0))
        || !(arguments.radial_min_radii < real_t(1))) {
        throw std::invalid_argument(
            "--radial-min-radii must lie strictly between zero and one"
        );
    }
    if (!(arguments.radial_max_radii > real_t(1))) {
        throw std::invalid_argument("--radial-max-radii must be greater than one");
    }
    return arguments;
}

std::vector<std::complex<real_t>> load_host_tensors(
    const fs::path& tensor_path,
    std::size_t face_count
) {
    const auto cuda_compatible = sie_mom::gstc::load_face_tensors_from_file(
        tensor_path.string(), face_count
    );
    std::vector<std::complex<real_t>> tensors;
    tensors.reserve(cuda_compatible.size());
    for (const auto& value : cuda_compatible) {
        tensors.emplace_back(value.x, value.y);
    }
    return tensors;
}

using FieldSample = sie_mom::ElectromagneticFieldSample;

struct ObservationSamples {
    std::vector<vec3_t> points;
    std::vector<int> regions;
    std::size_t zx_count = 0;
};

struct FieldRepresentations {
    std::vector<FieldSample> interior;
    std::vector<FieldSample> exterior;
    std::vector<FieldSample> physical;
};

int classify_region(const vec3_t& point, real_t sphere_radius) {
    const real_t radius = std::sqrt(
        point.x * point.x + point.y * point.y + point.z * point.z
    );
    return radius < sphere_radius ? 1 : 2;
}

ObservationSamples make_observation_samples(
    const Arguments& arguments,
    real_t sphere_radius
) {
    ObservationSamples samples;
    samples.zx_count = static_cast<std::size_t>(arguments.zx_grid_size)
        * static_cast<std::size_t>(arguments.zx_grid_size);
    const std::size_t total_count = samples.zx_count
        + static_cast<std::size_t>(arguments.radial_points);
    samples.points.reserve(total_count);
    samples.regions.reserve(total_count);

    const real_t span = arguments.zx_span_radii * sphere_radius;
    const real_t zx_denominator = static_cast<real_t>(arguments.zx_grid_size - 1);
    for (int z_index = 0; z_index < arguments.zx_grid_size; ++z_index) {
        const real_t z = -span
            + real_t(2) * span * static_cast<real_t>(z_index) / zx_denominator;
        for (int x_index = 0; x_index < arguments.zx_grid_size; ++x_index) {
            const real_t x = -span
                + real_t(2) * span * static_cast<real_t>(x_index) / zx_denominator;
            const vec3_t point = make_vec3(x, real_t(0), z);
            samples.points.push_back(point);
            samples.regions.push_back(classify_region(point, sphere_radius));
        }
    }

    const real_t pi = std::acos(real_t(-1));
    const real_t theta = arguments.radial_theta_degrees * pi / real_t(180);
    const real_t phi = arguments.radial_phi_degrees * pi / real_t(180);
    const real_t sin_theta = std::sin(theta);
    const real_t radial_denominator = static_cast<real_t>(arguments.radial_points - 1);
    for (int index = 0; index < arguments.radial_points; ++index) {
        const real_t normalized_radius = arguments.radial_min_radii
            + (arguments.radial_max_radii - arguments.radial_min_radii)
                * static_cast<real_t>(index) / radial_denominator;
        const real_t radius = normalized_radius * sphere_radius;
        const vec3_t point = make_vec3(
            radius * sin_theta * std::cos(phi),
            radius * sin_theta * std::sin(phi),
            radius * std::cos(theta)
        );
        samples.points.push_back(point);
        samples.regions.push_back(classify_region(point, sphere_radius));
    }
    return samples;
}

using sie_mom::current_block;
using sie_mom::subtract_fields;

FieldRepresentations evaluate_field_representations(
    const sie_mom::mesh::MeshData& mesh,
    const sie_mom::solvers::GstcSieOptions& options,
    const sie_mom::solvers::GstcSieCaseResult& result,
    const ObservationSamples& observations,
    const sie_mom::solvers::GstcExcitationCase& sources
) {
    const std::size_t edge_count = mesh.edges.size();
    if (result.physical_currents.size() != 4 * edge_count) {
        throw std::runtime_error(
            "physical current vector does not contain four RWG blocks"
        );
    }

    const auto j1 = current_block(result.physical_currents, 0, edge_count);
    const auto m1 = current_block(result.physical_currents, 1, edge_count);
    const auto j2 = current_block(result.physical_currents, 2, edge_count);
    const auto m2 = current_block(result.physical_currents, 3, edge_count);

    const auto interior_surface_integrals =
        sie_mom::solvers::evaluate_surface_integral_fields(
            mesh, j1, m1, observations.points,
            options.frequency_hz, options.interior, options.quadrature_rule,
            options.vacuum_permittivity, options.vacuum_permeability
        );
    const auto exterior_surface_integrals =
        sie_mom::solvers::evaluate_surface_integral_fields(
            mesh, j2, m2, observations.points,
            options.frequency_hz, options.exterior, options.quadrature_rule,
            options.vacuum_permittivity, options.vacuum_permeability
        );
    const auto interior_incident = sources.interior
        ? sie_mom::excitation::evaluate_incident_fields(observations.points, *sources.interior)
        : std::vector<FieldSample>(observations.points.size());
    const auto exterior_incident = sources.exterior
        ? sie_mom::excitation::evaluate_incident_fields(observations.points, *sources.exterior)
        : std::vector<FieldSample>(observations.points.size());

    FieldRepresentations fields;
    fields.interior.reserve(observations.points.size());
    fields.exterior.reserve(observations.points.size());
    fields.physical.reserve(observations.points.size());
    for (std::size_t index = 0; index < observations.points.size(); ++index) {
        fields.interior.push_back(
            subtract_fields(
                interior_incident[index],
                interior_surface_integrals[index]
            )
        );
        fields.exterior.push_back(
            subtract_fields(
                exterior_incident[index],
                exterior_surface_integrals[index]
            )
        );
        fields.physical.push_back(
            observations.regions[index] == 1
                ? fields.interior[index]
                : fields.exterior[index]
        );
    }
    return fields;
}

void write_field_csv(
    const fs::path& path,
    const ObservationSamples& observations,
    const std::vector<FieldSample>& fields,
    std::size_t offset,
    std::size_t count
) {
    if (observations.points.size() != observations.regions.size()
        || fields.size() != observations.points.size()
        || offset > fields.size()
        || count > fields.size() - offset) {
        throw std::invalid_argument("inconsistent field-output array sizes");
    }
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open field output: " + path.string());
    }
    stream << std::setprecision(17)
        << "point_index,x,y,z,region,"
        << "Ex_real,Ex_imag,Ey_real,Ey_imag,Ez_real,Ez_imag,"
        << "Hx_real,Hx_imag,Hy_real,Hy_imag,Hz_real,Hz_imag\n";
    for (std::size_t local_index = 0; local_index < count; ++local_index) {
        const std::size_t index = offset + local_index;
        const auto& point = observations.points[index];
        const auto& field = fields[index];
        stream << local_index << ','
            << point.x << ',' << point.y << ',' << point.z << ','
            << observations.regions[index];
        for (const auto& component : field.electric) {
            stream << ',' << component.real() << ',' << component.imag();
        }
        for (const auto& component : field.magnetic) {
            stream << ',' << component.real() << ',' << component.imag();
        }
        stream << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing field output: " + path.string());
    }
}

void write_field_outputs(
    const fs::path& output_directory,
    const ObservationSamples& observations,
    const FieldRepresentations& fields
) {
    const fs::path fields_directory = output_directory / "fields";
    fs::create_directories(fields_directory);
    const std::size_t radial_offset = observations.zx_count;
    const std::size_t radial_count = observations.points.size() - radial_offset;

    const std::array<std::pair<const char*, const std::vector<FieldSample>*>, 3>
        representations = {{
            {"interior_representation", &fields.interior},
            {"exterior_representation", &fields.exterior},
            {"physical", &fields.physical},
        }};
    for (const auto& representation : representations) {
        write_field_csv(
            fields_directory / (std::string("sie_") + representation.first + "_zx.csv"),
            observations,
            *representation.second,
            0,
            observations.zx_count
        );
        write_field_csv(
            fields_directory
                / (std::string("sie_") + representation.first + "_radial.csv"),
            observations,
            *representation.second,
            radial_offset,
            radial_count
        );
    }
}

void write_physical_currents(
    const fs::path& path,
    const std::vector<std::complex<real_t>>& currents
) {
    const std::uint16_t endian_probe = 1;
    const auto* first_byte = reinterpret_cast<const unsigned char*>(&endian_probe);
    if (*first_byte != 1) {
        throw std::runtime_error(
            "currents.bin export currently requires a little-endian host"
        );
    }

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open current output file: " + path.string());
    }

    std::vector<double> interleaved;
    interleaved.reserve(2 * currents.size());
    for (const auto& current : currents) {
        interleaved.push_back(static_cast<double>(current.real()));
        interleaved.push_back(static_cast<double>(current.imag()));
    }
    stream.write(
        reinterpret_cast<const char*>(interleaved.data()),
        static_cast<std::streamsize>(interleaved.size() * sizeof(double))
    );
    if (!stream) {
        throw std::runtime_error("failed while writing current output file: " + path.string());
    }
}

void write_solver_metrics(
    const fs::path& path,
    const std::string& case_id,
    const sie_mom::solvers::GstcSieCaseResult& result,
    sie_mom::GstcBasisType basis
) {
    const auto& metrics = result.metrics;
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open metrics output file: " + path.string());
    }

    stream << std::setprecision(17)
        << "{\n"
        << "  \"schema_version\": 2,\n"
        << "  \"artifact\": \"anisotropic_sphere_app_solution\",\n"
        << "  \"case_id\": \"" << case_id << "\",\n"
        << "  \"batch_metrics\": \"../../batch_metrics.json\",\n"
        << "  \"phasor_convention\": \"exp(+j omega t)\",\n"
        << "  \"scaling\": \"eta0\",\n"
        << "  \"basis\": \"" << basis_name(basis) << "\",\n"
        << "  \"current_data\": {\n"
        << "    \"path\": \"currents.bin\",\n"
        << "    \"ordering\": \"J1, M1, J2, M2; one block per RWG edge\",\n"
        << "    \"precision\": \"complex128\",\n"
        << "    \"byte_order\": \"little-endian\",\n"
        << "    \"complex_encoding\": \"interleaved real-imaginary float64\",\n"
        << "    \"physical_currents\": true,\n"
        << "    \"complex_value_count\": " << result.physical_currents.size() << "\n"
        << "  },\n"
        << "  \"solver\": {\n"
        << "    \"rhs_assembly_seconds\": " << metrics.rhs_assembly_seconds << ",\n"
        << "    \"residual_seconds\": " << metrics.residual_seconds << ",\n"
        << "    \"residual_computed\": "
        << (metrics.residual_computed ? "true" : "false") << ",\n"
        << "    \"scaled_rhs_norm\": " << metrics.rhs_norm << ",\n"
        << "    \"maximum_j1_coefficient\": "
        << metrics.maximum_j1_coefficient << ",\n"
        << "    \"maximum_m1_coefficient\": "
        << metrics.maximum_m1_coefficient << ",\n"
        << "    \"maximum_j2_coefficient\": "
        << metrics.maximum_j2_coefficient << ",\n"
        << "    \"maximum_m2_coefficient\": "
        << metrics.maximum_m2_coefficient;

    if (metrics.residual_computed) {
        stream
            << ",\n    \"scaled_absolute_residual\": "
            << metrics.absolute_residual
            << ",\n    \"scaled_relative_residual\": "
            << metrics.relative_residual << "\n";
    } else {
        stream << "\n";
    }
    stream << "  }\n}\n";

    if (!stream) {
        throw std::runtime_error("failed while writing metrics file: " + path.string());
    }
}

void write_batch_metrics(
    const fs::path& path,
    const sie_mom::solvers::GstcSieBatchMetrics& metrics,
    const std::vector<ExcitationCase>& cases,
    sie_mom::GstcBasisType basis
) {
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open batch metrics output: " + path.string());
    }
    stream << std::setprecision(17)
        << "{\n"
        << "  \"schema_version\": 2,\n"
        << "  \"artifact\": \"anisotropic_sphere_app_batch\",\n"
        << "  \"case_ids\": [";
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (index != 0) stream << ", ";
        stream << '"' << cases[index].id << '"';
    }
    stream << "],\n"
        << "  \"mesh\": {\n"
        << "    \"vertex_count\": " << metrics.vertex_count << ",\n"
        << "    \"face_count\": " << metrics.face_count << ",\n"
        << "    \"edge_count\": " << metrics.edge_count << "\n"
        << "  },\n"
        << "  \"solver\": {\n"
        << "    \"method\": \"dense_lu_batch\",\n"
        << "    \"scaling\": \"eta0\",\n"
        << "    \"basis\": \"" << basis_name(basis) << "\",\n"
        << "    \"case_count\": " << metrics.case_count << ",\n"
        << "    \"matrix_dimension\": " << metrics.matrix_dimension << ",\n"
        << "    \"dense_matrix_bytes\": " << metrics.dense_matrix_bytes << ",\n"
        << "    \"device_preparation_seconds\": " << metrics.device_preparation_seconds << ",\n"
        << "    \"matrix_assembly_seconds\": " << metrics.matrix_assembly_seconds << ",\n"
        << "    \"factorization_seconds\": " << metrics.factorization_seconds << ",\n"
        << "    \"solve_seconds\": " << metrics.solve_seconds << ",\n"
        << "    \"residual_matrix_assembly_seconds\": "
        << metrics.residual_matrix_assembly_seconds << "\n"
        << "  }\n}\n";
    if (!stream) {
        throw std::runtime_error("failed while writing batch metrics: " + path.string());
    }
}

void print_summary(
    const sie_mom::solvers::GstcSieBatchResult& result,
    sie_mom::GstcBasisType basis
) {
    const auto& metrics = result.metrics;
    std::cout << std::scientific << std::setprecision(6)
        << "Spherical anisotropic GSTC SIE\n"
        << "  basis: " << basis_name(basis) << "\n"
        << "  vertices: " << metrics.vertex_count << "\n"
        << "  faces: " << metrics.face_count << "\n"
        << "  edges: " << metrics.edge_count << "\n"
        << "  cases: " << metrics.case_count << "\n"
        << "  matrix_dimension: " << metrics.matrix_dimension << "\n"
        << "  dense_matrix_gib: "
        << static_cast<double>(metrics.dense_matrix_bytes)
            / (1024.0 * 1024.0 * 1024.0) << "\n"
        << "  device_preparation_seconds: "
        << metrics.device_preparation_seconds << "\n"
        << "  matrix_assembly_seconds: " << metrics.matrix_assembly_seconds << "\n"
        << "  factorization_seconds: " << metrics.factorization_seconds << "\n"
        << "  solve_seconds: " << metrics.solve_seconds << "\n"
        << "  residual_matrix_assembly_seconds: "
        << metrics.residual_matrix_assembly_seconds << "\n";
}

int run(const Arguments& arguments) {
    sie_mom::mesh::ObjMeshLoader mesh_loader(arguments.mesh_path.string());
    const auto& mesh = mesh_loader.getMeshData();
    const auto tensors = load_host_tensors(arguments.tensor_path, mesh.faces.size());

    const std::size_t matrix_dimension = 4 * mesh.edges.size();
    const long double dense_matrix_gib =
        static_cast<long double>(matrix_dimension)
        * static_cast<long double>(matrix_dimension)
        * static_cast<long double>(sizeof(complex__float))
        / (1024.0L * 1024.0L * 1024.0L);
    std::cout << "Loaded spherical GSTC SIE inputs\n"
        << "  basis: " << basis_name(arguments.basis) << "\n"
        << "  vertices: " << mesh.vertices.size() << "\n"
        << "  faces: " << mesh.faces.size() << "\n"
        << "  edges: " << mesh.edges.size() << "\n"
        << "  tensor_records: " << tensors.size() / 32 << "\n"
        << "  matrix_dimension: " << matrix_dimension << "\n"
        << "  dense_matrix_gib: " << std::fixed << std::setprecision(3)
        << static_cast<double>(dense_matrix_gib) << "\n"
        << "Starting dense CUDA solve...\n" << std::flush;

    sie_mom::solvers::GstcSieOptions options;
    options.frequency_hz = arguments.frequency_hz;
    options.interior.relative_permittivity = arguments.epsilon_r_interior;
    options.interior.relative_permeability = arguments.mu_r_interior;
    options.exterior.relative_permittivity = arguments.epsilon_r_exterior;
    options.exterior.relative_permeability = arguments.mu_r_exterior;

    options.quadrature_rule = arguments.quadrature_rule;

    options.singularity_threshold_lambda0 =
        arguments.singularity_threshold_lambda0;
    options.tensors_are_uniform = false;
    options.compute_residual = arguments.compute_residual;
    options.basis = arguments.basis;

    std::vector<sie_mom::solvers::GstcExcitationCase> sources;
    sources.reserve(arguments.cases.size());
    for (const auto& simulation_case : arguments.cases) {
        const auto prepare = [&](const auto& specification) {
            return specification
                ? std::optional<sie_mom::excitation::PreparedExcitation>(
                    sie_mom::excitation::prepare_source(
                        *specification, options.frequency_hz, options.interior, options.exterior,
                        options.vacuum_permittivity, options.vacuum_permeability
                    ))
                : std::nullopt;
        };
        sources.push_back({prepare(simulation_case.interior), prepare(simulation_case.exterior)});
    }
    std::optional<ObservationSamples> observations;
    if (arguments.evaluate_fields) {
        observations = make_observation_samples(arguments, sie_mom::mesh::infer_sphere_radius(mesh));
    }
    const auto result = sie_mom::solvers::solve_gstc_dense_batch(
        mesh, tensors, options, sources
    );

    fs::create_directories(arguments.output_directory);
    write_batch_metrics(
        arguments.output_directory / "batch_metrics.json",
        result.metrics,
        arguments.cases,
        arguments.basis
    );
    print_summary(result, arguments.basis);
    for (std::size_t index = 0; index < arguments.cases.size(); ++index) {
        const auto& case_id = arguments.cases[index].id;
        const auto& solution = result.cases.at(index);
        const fs::path output_directory = arguments.output_directory / "cases" / case_id;
        fs::create_directories(output_directory);
        write_physical_currents(output_directory / "currents.bin", solution.physical_currents);
        write_solver_metrics(
            output_directory / "solver_metrics.json", case_id, solution, arguments.basis
        );
        std::cout << "Case " << case_id << "\n"
            << "  outputs: " << output_directory << "\n";
        if (solution.metrics.residual_computed) {
            std::cout << "  scaled_relative_residual: " << solution.metrics.relative_residual << "\n";
        }
        if (observations) {
            const auto start = std::chrono::steady_clock::now();
            const auto fields = evaluate_field_representations(
                mesh, options, solution, *observations, sources[index]
            );
            write_field_outputs(output_directory, *observations, fields);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start
            ).count();
            std::cout << "  observation_points: " << observations->points.size() << "\n"
                << "  field_evaluation_and_output_seconds: " << seconds << "\n";
        }
    }
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    Arguments arguments;
    try {
        arguments = parse_arguments(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "anisotropic_sphere_app: " << error.what() << "\n\n";
        print_usage(argv[0]);
        return 2;
    }

    try {
        return run(arguments);
    } catch (const std::exception& error) {
        std::cerr << "anisotropic_sphere_app: " << error.what() << "\n";
        return 1;
    }
}
