// Sample regional integral contributions at offset centroids or supplied points.
// Python forms total fields as F_incident - Q; Q alone is not a total field.
#include "sie_mom/cuda/math_utils.cuh"
#include "sie_mom/mesh/mesh_utils.hpp"
#include "sie_mom/mesh/obj_mesh_loader.hpp"
#include "sie_mom/solvers/surface_current_fields.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Complex = std::complex<real_t>;

real_t positive_real(const char* text) {
    std::size_t consumed = 0;
    const std::string value(text);
    const double parsed = std::stod(value, &consumed);
    const real_t result = static_cast<real_t>(parsed);
    if (consumed != value.size() || !std::isfinite(result) || !(result > 0)) {
        throw std::invalid_argument("expected a finite positive value: " + value);
    }
    return result;
}

int quadrature_rule(const char* text) {
    std::size_t consumed = 0;
    const std::string value(text);
    const long result = std::stol(value, &consumed);
    if (consumed != value.size() || result < 1 || result > 20) {
        throw std::invalid_argument("quadrature rule must be an integer in [1, 20]");
    }
    return static_cast<int>(result);
}

void require_little_endian_host() {
    const std::uint16_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1) {
        throw std::runtime_error("binary input requires a little-endian host");
    }
}

std::vector<Complex> read_currents(const fs::path& path, std::size_t edge_count) {
    require_little_endian_host();
    const std::size_t value_count = 4 * edge_count;
    if (fs::file_size(path) != value_count * 2 * sizeof(double)) {
        throw std::runtime_error("currents.bin must contain four complex128 RWG blocks");
    }
    std::vector<double> values(2 * value_count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    std::vector<Complex> currents;
    currents.reserve(value_count);
    for (std::size_t index = 0; index < value_count; ++index) {
        if (!std::isfinite(values[2 * index]) || !std::isfinite(values[2 * index + 1])) {
            throw std::runtime_error("non-finite physical current coefficient");
        }
        currents.emplace_back(values[2 * index], values[2 * index + 1]);
    }
    return currents;
}

std::vector<vec3_t> read_points(const fs::path& path) {
    require_little_endian_host();
    const auto bytes = fs::file_size(path);
    if (bytes == 0 || bytes % (3 * sizeof(double)) != 0) {
        throw std::runtime_error("point input must contain nonempty float64 XYZ triples");
    }
    std::vector<double> values(bytes / sizeof(double));
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    std::vector<vec3_t> points;
    points.reserve(values.size() / 3);
    for (std::size_t index = 0; index < values.size(); index += 3) {
        const auto point = make_vec3(
            static_cast<real_t>(values[index]),
            static_cast<real_t>(values[index + 1]),
            static_cast<real_t>(values[index + 2])
        );
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            throw std::runtime_error("point input coordinates must be finite");
        }
        points.push_back(point);
    }
    return points;
}

void write_integrals(
    const fs::path& path,
    const std::vector<vec3_t>& points,
    const std::vector<sie_mom::ElectromagneticFieldSample>& fields,
    const char* index_name
) {
    if (fields.size() != points.size()) throw std::runtime_error("field count mismatch");
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    stream << std::setprecision(17)
           << index_name << ",x,y,z,QEx_real,QEx_imag,QEy_real,QEy_imag,QEz_real,QEz_imag,"
           << "QHx_real,QHx_imag,QHy_real,QHy_imag,QHz_real,QHz_imag\n";
    for (std::size_t face = 0; face < points.size(); ++face) {
        stream << face << ',' << points[face].x << ',' << points[face].y << ',' << points[face].z;
        for (const auto& field : {fields[face].electric, fields[face].magnetic}) {
            for (const auto& component : field) {
                if (!std::isfinite(component.real()) || !std::isfinite(component.imag())) {
                    throw std::runtime_error("non-finite sampled integral contribution");
                }
                stream << ',' << component.real() << ',' << component.imag();
            }
        }
        stream << '\n';
    }
    if (!stream) throw std::runtime_error("failed writing " + path.string());
}

void usage(const char* program) {
    std::cout << "Usage: " << program
              << " MESH CURRENTS OUTPUT_DIR FREQUENCY_HZ EPS_INT MU_INT EPS_EXT MU_EXT RULE OFFSET_M\n"
              << "Samples c +/- OFFSET_M * c/|c| in OBJ face order.\n"
              << "Replace OFFSET_M with --points POINTS_BIN to sample the supplied points in both regions.\n"
              << "POINTS_BIN contains little-endian float64 XYZ triples; CSV IDs are point_index.\n"
              << "Input currents are physical [J1,M1,J2,M2]; output is Q, not incident - Q.\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        usage(argv[0]);
        return 0;
    }
    const bool supplied_points = argc == 12 && std::string(argv[10]) == "--points";
    if (argc != 11 && !supplied_points) {
        usage(argv[0]);
        return 2;
    }
    try {
        const real_t frequency = positive_real(argv[4]);
        sie_mom::IsotropicMedium interior, exterior;
        interior.relative_permittivity = positive_real(argv[5]);
        interior.relative_permeability = positive_real(argv[6]);
        exterior.relative_permittivity = positive_real(argv[7]);
        exterior.relative_permeability = positive_real(argv[8]);
        const int rule = quadrature_rule(argv[9]);
        const real_t offset = supplied_points ? real_t(0) : positive_real(argv[10]);
        const auto input_points = supplied_points ? read_points(argv[11]) : std::vector<vec3_t>{};
        const sie_mom::mesh::ObjMeshLoader loader(argv[1]);
        const auto& mesh = loader.getMeshData();
        const auto currents = read_currents(argv[2], mesh.edges.size());
        const fs::path output(argv[3]);

        for (std::size_t region = 0; region < 2; ++region) {
            const auto points = supplied_points
                ? input_points : sie_mom::mesh::offset_points(mesh, region == 0 ? -offset : offset);
            const auto first = currents.begin() + static_cast<std::ptrdiff_t>(2 * region * mesh.edges.size());
            const std::vector<Complex> electric(first, first + mesh.edges.size());
            const std::vector<Complex> magnetic(first + mesh.edges.size(), first + 2 * mesh.edges.size());
            const auto fields = sie_mom::solvers::evaluate_surface_integral_fields(
                mesh, electric, magnetic, points, frequency,
                region == 0 ? interior : exterior, rule
            );
            fs::create_directories(output);
            write_integrals(output / (region == 0 ? "interior_integrals.csv" : "exterior_integrals.csv"),
                            points, fields, supplied_points ? "point_index" : "face_index");
        }
        if (supplied_points) {
            std::cout << "Sampled " << input_points.size() << " supplied points in each region\n";
        } else {
            std::cout << "Sampled " << mesh.faces.size() << " faces on each side; offset=" << offset << " m\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "anisotropic_surface_sampler: " << error.what() << '\n';
        return 1;
    }
}
