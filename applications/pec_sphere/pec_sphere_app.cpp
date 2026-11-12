#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/mesh/obj_mesh_loader.hpp"
#include "sie_mom/solvers/pec_solver.hpp"
#include "sie_mom/solvers/surface_current_fields.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Current = std::complex<real_t>;
using Field = sie_mom::ElectromagneticFieldSample;

struct Arguments {
    fs::path mesh_path;
    fs::path output_directory;
    fs::path currents_file;
    sie_mom::solvers::PecOptions options;
    bool evaluate_fields = true;
    int zx_grid_size = 80;
    real_t zx_span_radii = real_t(2);
    int radial_points = 100;
    real_t radial_theta_degrees = real_t(45);
    real_t radial_phi_degrees = real_t(0);
    real_t radial_min_radii = real_t(0.05);
    real_t radial_max_radii = real_t(2);
};

void usage(const char* name) {
    std::cout << "Usage: " << name << " --mesh PATH --output-dir PATH --frequency-hz HZ [options]\n"
              << "  --epsilon-r-exterior V          Default: 1\n"
              << "  --mu-r-exterior V               Default: 1\n"
              << "  --formulation [t-efie|n-efie|n-mfie]  Default: t-efie; t/n aliases accepted\n"
              << "  --ntl-form [surface|line]       N-EFIE only; Form B or A, default: surface\n"
              << "  --line-quadrature-points N      N-EFIE Form A only, default: 20\n"
              << "  --quadrature-rule N             Dunavant rule, default: 7\n"
              << "  --singularity-threshold-lambda0 V  Default: 0.1\n"
              << "  --no-residual                   Skip the matrix residual\n"
              << "  --export-matrix                 Save the pre-LU complex matrix as matrix.npy\n"
              << "  --currents-file PATH            Evaluate saved J2 without solving again\n"
              << "  --no-fields                     Skip field evaluation\n"
              << "  --zx-grid-size N                ZX points per axis, default: 80; 0 skips grid\n"
              << "  --zx-span-radii V               ZX half-span in sphere radii, default: 2\n"
              << "  --radial-points N               Radial samples, default: 100; 0 skips cut\n"
              << "  --radial-theta-degrees V        Radial theta, default: 45\n"
              << "  --radial-phi-degrees V          Radial phi, default: 0\n"
              << "  --radial-min-radii V            Minimum r/a, default: 0.05\n"
              << "  --radial-max-radii V            Maximum r/a, default: 2\n"
              << "Source: unit x-polarized plane wave traveling +z in region 2.\n";
}

std::string value_after(int argc, char* argv[], int& index) {
    if (++index >= argc) throw std::invalid_argument("missing option value");
    return argv[index];
}

Arguments parse(int argc, char* argv[]) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (option == "--mesh") {
            args.mesh_path = value_after(argc, argv, i);
        } else if (option == "--output-dir") {
            args.output_directory = value_after(argc, argv, i);
        } else if (option == "--frequency-hz") {
            args.options.frequency_hz = std::stod(value_after(argc, argv, i));
        } else if (option == "--epsilon-r-exterior") {
            args.options.exterior.relative_permittivity = std::stod(value_after(argc, argv, i));
        } else if (option == "--mu-r-exterior") {
            args.options.exterior.relative_permeability = std::stod(value_after(argc, argv, i));
        } else if (option == "--formulation") {
            const auto value = value_after(argc, argv, i);
            if (value == "t" || value == "t-efie")
                args.options.formulation = sie_mom::solvers::PecFormulation::TEfie;
            else if (value == "n" || value == "n-efie")
                args.options.formulation = sie_mom::solvers::PecFormulation::NEfie;
            else if (value == "n-mfie")
                args.options.formulation = sie_mom::solvers::PecFormulation::NMfie;
            else throw std::invalid_argument(
                "--formulation must be t-efie, n-efie, or n-mfie");
        } else if (option == "--ntl-form") {
            const auto value = value_after(argc, argv, i);
            if (value == "surface") args.options.ntl_form = sie_mom::solvers::PecNtlForm::Surface;
            else if (value == "line") args.options.ntl_form = sie_mom::solvers::PecNtlForm::Line;
            else throw std::invalid_argument("--ntl-form must be surface or line");
        } else if (option == "--line-quadrature-points") {
            args.options.line_quadrature_points = std::stoi(value_after(argc, argv, i));
        } else if (option == "--quadrature-rule") {
            args.options.quadrature_rule = std::stoi(value_after(argc, argv, i));
        } else if (option == "--singularity-threshold-lambda0") {
            args.options.singularity_threshold_lambda0 = std::stod(value_after(argc, argv, i));
        } else if (option == "--no-residual") {
            args.options.compute_residual = false;
        } else if (option == "--export-matrix") {
            args.options.capture_assembled_matrix = true;
        } else if (option == "--currents-file") {
            args.currents_file = value_after(argc, argv, i);
        } else if (option == "--no-fields") {
            args.evaluate_fields = false;
        } else if (option == "--zx-grid-size") {
            args.zx_grid_size = std::stoi(value_after(argc, argv, i));
        } else if (option == "--zx-span-radii") {
            args.zx_span_radii = std::stod(value_after(argc, argv, i));
        } else if (option == "--radial-points") {
            args.radial_points = std::stoi(value_after(argc, argv, i));
        } else if (option == "--radial-theta-degrees") {
            args.radial_theta_degrees = std::stod(value_after(argc, argv, i));
        } else if (option == "--radial-phi-degrees") {
            args.radial_phi_degrees = std::stod(value_after(argc, argv, i));
        } else if (option == "--radial-min-radii") {
            args.radial_min_radii = std::stod(value_after(argc, argv, i));
        } else if (option == "--radial-max-radii") {
            args.radial_max_radii = std::stod(value_after(argc, argv, i));
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (args.mesh_path.empty() || args.output_directory.empty() ||
        !(args.options.frequency_hz > real_t(0)))
        throw std::invalid_argument("--mesh, --output-dir, and positive --frequency-hz are required");
    if (args.zx_grid_size < 0 || args.zx_grid_size == 1 ||
        args.radial_points < 0 || args.radial_points == 1 ||
        !(args.zx_span_radii > real_t(0)) ||
        !(args.radial_min_radii >= real_t(0)) ||
        !(args.radial_max_radii > args.radial_min_radii))
        throw std::invalid_argument("invalid field sampling range or point count");
    if (!args.currents_file.empty() && !args.evaluate_fields)
        throw std::invalid_argument("--currents-file requires field evaluation");
    if (!args.currents_file.empty() && args.options.capture_assembled_matrix)
        throw std::invalid_argument("--export-matrix requires solving, not --currents-file");
    return args;
}

void write_matrix_npy(const fs::path& path, const std::vector<Current>& matrix,
                      std::size_t dimension) {
    if (matrix.size() != dimension * dimension)
        throw std::runtime_error("assembled PEC matrix has the wrong dimension");
    static_assert(sizeof(Current) == 2 * sizeof(real_t),
                  "NumPy complex storage requires two adjacent real components");
    const char* dtype = sizeof(real_t) == 8 ? "<c16" : "<c8";
    std::string header = "{'descr': '" + std::string(dtype) +
        "', 'fortran_order': True, 'shape': (" + std::to_string(dimension) +
        ", " + std::to_string(dimension) + "), }";
    while ((10 + header.size() + 1) % 16 != 0) header += ' ';
    header += '\n';
    if (header.size() > 65535)
        throw std::runtime_error("NumPy matrix header is too long");
    const std::uint16_t header_size = static_cast<std::uint16_t>(header.size());
    const char header_length[2] = {
        static_cast<char>(header_size & 0xff),
        static_cast<char>((header_size >> 8) & 0xff)
    };
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open " + path.string());
    output.write("\x93NUMPY", 6);
    const char version[2] = {1, 0};
    output.write(version, 2);
    output.write(header_length, 2);
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(matrix.data()),
                 static_cast<std::streamsize>(matrix.size() * sizeof(Current)));
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

real_t sphere_radius(const sie_mom::mesh::MeshData& mesh) {
    real_t sum = real_t(0);
    for (const auto& vertex : mesh.vertices)
        sum += std::sqrt(vertex.x * vertex.x + vertex.y * vertex.y + vertex.z * vertex.z);
    return sum / static_cast<real_t>(mesh.vertices.size());
}

struct Observations {
    std::vector<vec3_t> points;
    std::vector<int> regions;
    std::size_t zx_count = 0;
};

Observations make_observations(const Arguments& args, real_t radius) {
    Observations result;
    if (args.zx_grid_size > 0) {
        const real_t span = args.zx_span_radii * radius;
        for (int iz = 0; iz < args.zx_grid_size; ++iz) {
            const real_t z = span * (real_t(2) * iz / (args.zx_grid_size - 1) - real_t(1));
            for (int ix = 0; ix < args.zx_grid_size; ++ix) {
                const real_t x = span * (real_t(2) * ix / (args.zx_grid_size - 1) - real_t(1));
                result.points.push_back(make_vec3(x, real_t(0), z));
                result.regions.push_back(std::hypot(x, z) < radius ? 1 : 2);
            }
        }
        result.zx_count = result.points.size();
    }
    if (args.radial_points > 0) {
        const real_t pi = std::acos(real_t(-1));
        const real_t theta = args.radial_theta_degrees * pi / real_t(180);
        const real_t phi = args.radial_phi_degrees * pi / real_t(180);
        for (int step = 0; step < args.radial_points; ++step) {
            const real_t fraction = static_cast<real_t>(step) / (args.radial_points - 1);
            const real_t r = radius * (args.radial_min_radii +
                fraction * (args.radial_max_radii - args.radial_min_radii));
            const real_t x = r * std::sin(theta) * std::cos(phi);
            const real_t y = r * std::sin(theta) * std::sin(phi);
            const real_t z = r * std::cos(theta);
            result.points.push_back(make_vec3(x, y, z));
            result.regions.push_back(r < radius ? 1 : 2);
        }
    }
    return result;
}

std::vector<Current> read_currents(const fs::path& path, std::size_t edge_count) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    std::string line;
    if (!std::getline(input, line) || line != "edge,J2_real,J2_imag,J2_abs")
        throw std::runtime_error("unexpected current CSV header: " + path.string());
    std::vector<Current> currents;
    while (std::getline(input, line)) {
        const auto a = line.find(',');
        const auto b = a == std::string::npos ? a : line.find(',', a + 1);
        const auto c = b == std::string::npos ? b : line.find(',', b + 1);
        if (a == std::string::npos || b == std::string::npos || c == std::string::npos ||
            std::stoull(line.substr(0, a)) != currents.size())
            throw std::runtime_error("invalid current CSV row: " + line);
        const auto re = std::stod(line.substr(a + 1, b - a - 1));
        const auto im = std::stod(line.substr(b + 1, c - b - 1));
        if (!std::isfinite(re) || !std::isfinite(im))
            throw std::runtime_error("non-finite current CSV row: " + line);
        currents.emplace_back(static_cast<real_t>(re), static_cast<real_t>(im));
    }
    if (currents.size() != edge_count)
        throw std::runtime_error("current CSV edge count does not match mesh");
    return currents;
}

void write_field_csv(const fs::path& path, const Observations& observations,
                     const std::vector<Field>& fields, std::size_t begin, std::size_t end) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot open " + path.string());
    output << std::scientific << std::setprecision(12)
           << "x,y,z,region,E_x_re,E_x_im,E_y_re,E_y_im,E_z_re,E_z_im,"
              "H_x_re,H_x_im,H_y_re,H_y_im,H_z_re,H_z_im\n";
    for (std::size_t i = begin; i < end; ++i) {
        const auto& p = observations.points[i];
        const auto& f = fields[i];
        output << p.x << ',' << p.y << ',' << p.z << ',' << observations.regions[i];
        for (const auto& e : f.electric) output << ',' << e.real() << ',' << e.imag();
        for (const auto& h : f.magnetic) output << ',' << h.real() << ',' << h.imag();
        output << '\n';
    }
    if (!output) throw std::runtime_error("failed writing " + path.string());
}

void evaluate_fields(const sie_mom::mesh::MeshData& mesh,
                     const std::vector<Current>& j2, const Arguments& args,
                     const sie_mom::excitation::PreparedExcitation& source) {
    const auto radius = sphere_radius(mesh);
    const auto observations = make_observations(args, radius);
    std::cout << "Evaluating " << observations.points.size()
              << " field points inside and outside the sphere...\n" << std::flush;
    const std::vector<Current> zero_m(mesh.edges.size(), Current{});
    const auto integrals = sie_mom::solvers::evaluate_surface_integral_fields(
        mesh, j2, zero_m, observations.points, args.options.frequency_hz,
        args.options.exterior, args.options.quadrature_rule,
        args.options.vacuum_permittivity, args.options.vacuum_permeability);
    const auto incident = sie_mom::excitation::evaluate_incident_fields(
        observations.points, source);
    std::vector<Field> fields(observations.points.size());
    for (std::size_t i = 0; i < fields.size(); ++i) {
        for (int component = 0; component < 3; ++component) {
            fields[i].electric[component] = incident[i].electric[component] -
                integrals[i].electric[component];
            fields[i].magnetic[component] = incident[i].magnetic[component] -
                integrals[i].magnetic[component];
        }
    }
    if (observations.zx_count > 0) {
        const auto path = args.output_directory / "zx_fields.csv";
        write_field_csv(path, observations, fields, 0, observations.zx_count);
        std::cout << "zx_fields_csv=" << path << '\n';
    }
    if (observations.points.size() > observations.zx_count) {
        const auto path = args.output_directory / "radial_fields.csv";
        write_field_csv(path, observations, fields,
                        observations.zx_count, observations.points.size());
        std::cout << "radial_fields_csv=" << path << '\n';
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const auto args = parse(argc, argv);
        sie_mom::mesh::ObjMeshLoader loader(args.mesh_path.string());
        const auto& mesh = loader.getMeshData();

        sie_mom::excitation::PlaneWaveExcitation wave;
        wave.electric_amplitude = make_vec3(real_t(1), real_t(0), real_t(0));
        wave.propagation_direction = make_vec3(real_t(0), real_t(0), real_t(1));
        wave.region = sie_mom::excitation::SourceRegion::exterior;
        const auto source = sie_mom::excitation::prepare_source(
            wave, args.options.frequency_hz,
            args.options.exterior, args.options.exterior,
            args.options.vacuum_permittivity, args.options.vacuum_permeability);

        fs::create_directories(args.output_directory);
        if (!args.currents_file.empty()) {
            const auto currents = read_currents(args.currents_file, mesh.edges.size());
            evaluate_fields(mesh, currents, args, source);
            return 0;
        }

        const auto formulation = args.options.formulation;
        if (formulation == sie_mom::solvers::PecFormulation::NMfie)
            std::cout << "N-MFIE: (1/2 M - nTK2)(J2) = <f x n, H_inc,2>\n";
        else if (formulation == sie_mom::solvers::PecFormulation::NEfie)
            std::cout << "N-EFIE: i*omega*mu2*nTL2(J2) = <f x n, E_inc,2>\n";
        else
            std::cout << "T-EFIE: i*omega*mu2*TL2(J2) = <f, E_inc,2>\n";
        std::cout << "mesh=" << args.mesh_path << " faces=" << mesh.faces.size()
                  << " edges=" << mesh.edges.size() << '\n'
                  << "quadrature_rule=" << args.options.quadrature_rule
                  << " singularity_threshold_lambda0="
                  << args.options.singularity_threshold_lambda0 << '\n';
        if (formulation == sie_mom::solvers::PecFormulation::NMfie)
            std::cout << "operator_quadrature=Form A nTK (zero 1D points)\n";
        if (formulation == sie_mom::solvers::PecFormulation::NEfie)
            std::cout << "ntl_form="
                      << (args.options.ntl_form == sie_mom::solvers::PecNtlForm::Line
                          ? "line (Form A)" : "surface (Form B)") << '\n';
        const auto result = sie_mom::solvers::solve_pec_dense(
            mesh, args.options, source);
        std::cout << std::setprecision(9)
                  << "matrix_dimension=" << result.metrics.matrix_dimension << '\n'
                  << "assembly_seconds=" << result.metrics.assembly_seconds << '\n'
                  << "solve_seconds=" << result.metrics.solve_seconds << '\n'
                  << "maximum_abs_J2=" << result.metrics.maximum_j_coefficient << '\n';
        if (result.metrics.residual_computed)
            std::cout << "relative_residual=" << result.metrics.relative_residual << '\n';

        if (args.options.capture_assembled_matrix) {
            const auto matrix_path = args.output_directory / "matrix.npy";
            write_matrix_npy(matrix_path, result.assembled_matrix,
                             result.metrics.matrix_dimension);
            std::cout << "matrix_npy=" << matrix_path << '\n';
            const auto edge_path = args.output_directory / "matrix_edges.csv";
            std::ofstream edge_output(edge_path);
            if (!edge_output)
                throw std::runtime_error("cannot open " + edge_path.string());
            edge_output << "edge,vertex_a,vertex_b\n";
            for (std::size_t edge = 0; edge < mesh.edges.size(); ++edge)
                edge_output << edge << ',' << mesh.edges[edge].x << ','
                            << mesh.edges[edge].y << '\n';
            if (!edge_output)
                throw std::runtime_error("failed writing " + edge_path.string());
            std::cout << "matrix_edges_csv=" << edge_path << '\n';
        }

        const auto path = args.output_directory / "currents.csv";
        std::ofstream output(path);
        if (!output) throw std::runtime_error("cannot open " + path.string());
        output << "edge,J2_real,J2_imag,J2_abs\n" << std::setprecision(17);
        for (std::size_t edge = 0; edge < result.exterior_current.size(); ++edge) {
            const auto value = result.exterior_current[edge];
            output << edge << ',' << value.real() << ',' << value.imag() << ','
                   << std::abs(value) << '\n';
        }
        if (!output) throw std::runtime_error("failed writing " + path.string());
        output.close();
        std::cout << "currents_csv=" << path << '\n';
        if (args.evaluate_fields)
            evaluate_fields(mesh, result.exterior_current, args, source);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
