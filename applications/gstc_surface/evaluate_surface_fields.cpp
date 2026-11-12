// Evaluate physical regional E/H from saved currents and classified points.
// This executable has no geometry classifier, tensor extraction or linear solve.
#include "sie_mom/excitation/excitation.hpp"
#include "sie_mom/mesh/obj_mesh_loader.hpp"
#include "sie_mom/solvers/surface_current_fields.hpp"
#include "sie_mom/utils/cli_helpers.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Complex = std::complex<real_t>;
using Field = sie_mom::ElectromagneticFieldSample;
using namespace sie_mom::cli;

struct Arguments {
    fs::path mesh, currents, observations, output;
    real_t frequency = 0;
    std::array<sie_mom::IsotropicMedium, 2> media{{{0, 0}, {0, 0}}};
    std::array<std::optional<sie_mom::excitation::SourceSpec>, 2> sources;
    int quadrature = 7;
};

Arguments parse_arguments(int argc, char* argv[]) {
    Arguments args;
    std::map<std::string, real_t*> numbers{
        {"--frequency-hz", &args.frequency},
        {"--epsilon-r-interior", &args.media[0].relative_permittivity},
        {"--mu-r-interior", &args.media[0].relative_permeability},
        {"--epsilon-r-exterior", &args.media[1].relative_permittivity},
        {"--mu-r-exterior", &args.media[1].relative_permeability},
    };
    std::map<std::string, fs::path*> paths{
        {"--mesh", &args.mesh}, {"--currents", &args.currents},
        {"--observations", &args.observations}, {"--output", &args.output},
    };
    bool case_seen = false;
    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        if (numbers.count(option)) {
            *numbers.at(option) = parse_real(require_value(argc, argv, index), option);
        } else if (paths.count(option)) {
            *paths.at(option) = require_value(argc, argv, index);
        } else if (option == "--quadrature-rule") {
            args.quadrature = parse_integer(require_value(argc, argv, index), option);
        } else if (option == "--case") {
            if (case_seen) throw std::invalid_argument("evaluate one current case per invocation");
            require_value(argc, argv, index); // Label only; case selection belongs to Python.
            case_seen = true;
        } else if (option == "--dipole" || option == "--plane-wave") {
            const auto region = parse_region(require_value(argc, argv, index));
            const auto slot = static_cast<std::size_t>(region) - 1;
            if (args.sources[slot]) throw std::invalid_argument("duplicate source region");
            if (option == "--dipole") {
                const std::string type = require_value(argc, argv, index);
                if (type != "electric" && type != "magnetic") {
                    throw std::invalid_argument("dipole type must be electric or magnetic");
                }
                sie_mom::excitation::DipoleExcitation source;
                source.region = region;
                source.is_magnetic = type == "magnetic";
                source.effective_moment = parse_vector(argc, argv, index, option);
                source.position = parse_vector(argc, argv, index, option);
                args.sources[slot] = source;
            } else {
                sie_mom::excitation::PlaneWaveExcitation source;
                source.region = region;
                source.electric_amplitude = parse_vector(argc, argv, index, option);
                source.propagation_direction = parse_vector(argc, argv, index, option);
                args.sources[slot] = source;
            }
        } else if (option == "--help" || option == "-h") {
            std::cout
                << "Usage: evaluate_surface_fields --mesh OBJ --currents BIN "
                << "--observations CSV --output CSV --frequency-hz F "
                << "--epsilon-r-interior ER --mu-r-interior MR "
                << "--epsilon-r-exterior ER --mu-r-exterior MR\n"
                << "Optional: --quadrature-rule N; --case ID; "
                << "--dipole REGION TYPE PX PY PZ X Y Z; "
                << "--plane-wave REGION EX EY EZ DX DY DZ\n"
                << "Observations: point_index,x,y,z,region (1=interior, 2=exterior).\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    for (const auto& [name, value] : paths) {
        if (value->empty()) throw std::invalid_argument(name + " is required");
    }
    for (const auto& [name, value] : numbers) {
        if (!(*value > 0)) throw std::invalid_argument(name + " must be positive");
    }
    if (args.quadrature < 1 || args.quadrature > 20) {
        throw std::invalid_argument("quadrature-rule must be in [1, 20]");
    }
    if (fs::exists(args.output)) throw std::invalid_argument("output already exists");
    return args;
}

std::vector<Complex> read_currents(const fs::path& path, std::size_t edges) {
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1) {
        throw std::runtime_error("current binary requires a little-endian host");
    }
    if (fs::file_size(path) != 4 * edges * 2 * sizeof(double)) {
        throw std::runtime_error("currents must contain four complex128 RWG blocks");
    }
    std::vector<double> raw(8 * edges);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(raw.data()),
                static_cast<std::streamsize>(raw.size() * sizeof(double)));
    if (!stream) throw std::runtime_error("cannot read physical currents");
    std::vector<Complex> currents;
    for (std::size_t i = 0; i < raw.size(); i += 2) {
        const Complex value(static_cast<real_t>(raw[i]), static_cast<real_t>(raw[i + 1]));
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            throw std::runtime_error("non-finite physical current coefficient");
        }
        currents.push_back(value);
    }
    return currents;
}

struct Observations {
    std::vector<vec3_t> points;
    std::vector<int> regions;
};

Observations read_observations(const fs::path& path) {
    std::ifstream stream(path);
    std::string line;
    if (!std::getline(stream, line)) throw std::runtime_error("cannot read observations");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "point_index,x,y,z,region") {
        throw std::runtime_error("expected observation header point_index,x,y,z,region");
    }
    Observations result;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::stringstream row(line);
        std::vector<std::string> cells;
        std::string cell;
        while (std::getline(row, cell, ',')) cells.push_back(cell);
        if (cells.size() != 5) throw std::runtime_error("expected five observation columns");
        const int index = parse_integer(cells[0].c_str(), "point_index");
        const int region = parse_integer(cells[4].c_str(), "region");
        if (index < 0 || static_cast<std::size_t>(index) != result.points.size()) {
            throw std::runtime_error("observation indices must be ordered and contiguous from zero");
        }
        if (region != 1 && region != 2) throw std::runtime_error("region must be 1 or 2");
        result.points.push_back(make_vec3(
            parse_real(cells[1].c_str(), "x"), parse_real(cells[2].c_str(), "y"),
            parse_real(cells[3].c_str(), "z")
        ));
        result.regions.push_back(region);
    }
    if (result.points.empty()) throw std::runtime_error("observations must not be empty");
    return result;
}

std::vector<Field> evaluate(
    const Arguments& args, const sie_mom::mesh::MeshData& mesh,
    const std::vector<Complex>& currents, const Observations& observations
) {
    std::vector<Field> result(observations.points.size());
    for (std::size_t region = 0; region < 2; ++region) {
        std::vector<std::size_t> indices;
        std::vector<vec3_t> points;
        for (std::size_t i = 0; i < observations.points.size(); ++i) {
            if (observations.regions[i] == static_cast<int>(region + 1)) {
                indices.push_back(i);
                points.push_back(observations.points[i]);
            }
        }
        if (points.empty()) continue;
        const auto integral = sie_mom::solvers::evaluate_surface_integral_fields(
            mesh, sie_mom::current_block(currents, 2 * region, mesh.edges.size()),
            sie_mom::current_block(currents, 2 * region + 1, mesh.edges.size()),
            points, args.frequency, args.media[region], args.quadrature
        );
        std::vector<Field> incident(points.size());
        if (args.sources[region]) {
            const auto source = sie_mom::excitation::prepare_source(
                *args.sources[region], args.frequency, args.media[0], args.media[1]
            );
            incident = sie_mom::excitation::evaluate_incident_fields(points, source);
        }
        for (std::size_t i = 0; i < points.size(); ++i) {
            const auto total = sie_mom::subtract_fields(incident[i], integral[i]);
            for (const auto* vector : {&total.electric, &total.magnetic}) {
                for (const auto& value : *vector) {
                    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
                        throw std::runtime_error("non-finite physical field");
                    }
                }
            }
            result[indices[i]] = total;
        }
    }
    return result;
}

void write_fields(const fs::path& path, const Observations& observations,
                  const std::vector<Field>& fields) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot open field output");
    stream << std::setprecision(17)
        << "point_index,x,y,z,region,Ex_real,Ex_imag,Ey_real,Ey_imag,Ez_real,Ez_imag,"
        << "Hx_real,Hx_imag,Hy_real,Hy_imag,Hz_real,Hz_imag\n";
    for (std::size_t i = 0; i < fields.size(); ++i) {
        const auto& point = observations.points[i];
        stream << i << ',' << point.x << ',' << point.y << ',' << point.z << ','
               << observations.regions[i];
        for (const auto* vector : {&fields[i].electric, &fields[i].magnetic}) {
            for (const auto& value : *vector) stream << ',' << value.real() << ',' << value.imag();
        }
        stream << '\n';
    }
    if (!stream) throw std::runtime_error("failed while writing field output");
}
} // namespace

int main(int argc, char* argv[]) {
    try {
        const auto args = parse_arguments(argc, argv);
        const sie_mom::mesh::ObjMeshLoader loader(args.mesh.string());
        const auto& mesh = loader.getMeshData();
        const auto currents = read_currents(args.currents, mesh.edges.size());
        const auto observations = read_observations(args.observations);
        write_fields(args.output, observations, evaluate(args, mesh, currents, observations));
        std::cout << "Evaluated " << observations.points.size()
                  << " physical E/H samples from saved currents\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "evaluate_surface_fields: " << error.what() << '\n';
        return 1;
    }
}
