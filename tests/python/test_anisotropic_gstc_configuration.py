import json
from dataclasses import FrozenInstanceError, replace

import pytest

from applications.anisotropic_gstc_sphere.workflow import (
    DipoleFieldDefinition,
    FieldReference,
    GSTCFieldsConfiguration,
    GSTCTransformationConfiguration,
    PlaneWaveFieldDefinition,
    SimulationCaseConfiguration,
    SimulationConfiguration,
    SimulationExcitationConfiguration,
    SphereCase,
    WorkflowConfiguration,
    load_workflow_configuration,
    write_json,
)


@pytest.mark.parametrize(
    "electric, direction",
    [
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 1j, 0.0), (0.0, 0.0, 2.0)),
        ((1.0 + 2j, -1.0 - 2j, 0.0), (1.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((1.0, 0.0, 5e-13), (0.0, 0.0, 1.0)),
        ((1e6, 0.0, 5e-7), (0.0, 0.0, 1.0)),
    ],
)
def test_plane_wave_preserves_complex_amplitudes_and_nonunit_directions(electric, direction):
    definition = PlaneWaveFieldDefinition(electric, direction)
    assert definition.electric_amplitude == electric
    assert all(isinstance(component, complex) for component in definition.electric_amplitude)
    assert definition.propagation_direction == direction


def test_plane_wave_is_immutable_even_when_constructed_from_lists():
    electric = [1.0, 1j, 0.0]
    direction = [0.0, 0.0, 2.0]
    definition = PlaneWaveFieldDefinition(electric, direction)
    electric[0] = 5.0
    direction[2] = 5.0
    assert definition.electric_amplitude == (1.0 + 0j, 1j, 0j)
    assert definition.propagation_direction == (0.0, 0.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        definition.electric_amplitude = (2.0 + 0j, 0j, 0j)


@pytest.mark.parametrize(
    "electric, direction, message",
    [
        ((1.0, 0.0), (0.0, 0.0, 1.0), "electric_amplitude.*three"),
        ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), "electric_amplitude.*three"),
        ("abc", (0.0, 0.0, 1.0), "electric_amplitude.*three"),
        ((True, 0.0, 0.0), (0.0, 0.0, 1.0), "electric_amplitude.*finite"),
        (("1", 0.0, 0.0), (0.0, 0.0, 1.0), "electric_amplitude.*finite"),
        ((float("nan"), 0.0, 0.0), (0.0, 0.0, 1.0), "electric_amplitude.*finite"),
        ((float("inf"), 0.0, 0.0), (0.0, 0.0, 1.0), "electric_amplitude.*finite"),
        ((complex(1.0, float("nan")), 0.0, 0.0), (0.0, 0.0, 1.0), "electric_amplitude.*finite"),
        ((complex(1.0, float("inf")), 0.0, 0.0), (0.0, 0.0, 1.0), "electric_amplitude.*finite"),
        ((1.0, 0.0, 0.0), (0.0, 1.0), "propagation_direction.*three"),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), "propagation_direction.*three"),
        ((1.0, 0.0, 0.0), None, "propagation_direction.*three"),
        ((1.0, 0.0, 0.0), (0.0, False, 1.0), "propagation_direction.*finite"),
        ((1.0, 0.0, 0.0), (0.0, 0.0, "1"), "propagation_direction.*finite"),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0 + 0j), "propagation_direction.*finite"),
        ((1.0, 0.0, 0.0), (0.0, 0.0, float("nan")), "propagation_direction.*finite"),
        ((1.0, 0.0, 0.0), (0.0, 0.0, float("inf")), "propagation_direction.*finite"),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), "propagation_direction.*nonzero"),
        ((1.0, 0.0, 1.0), (0.0, 0.0, 1.0), "transverse"),
        ((1.0, 0.0, 1j), (0.0, 0.0, 1.0), "transverse"),
    ],
)
def test_plane_wave_rejects_invalid_parameters(electric, direction, message):
    with pytest.raises(ValueError, match=message):
        PlaneWaveFieldDefinition(electric, direction)


@pytest.mark.parametrize("dipole_type", ["electric", "magnetic"])
@pytest.mark.parametrize(
    "moment, position",
    [
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
        ((1.0 + 2j, -1j, 3.0), (-0.2, 0.3, 0.4)),
        ((0.0, 0.0, 0.0), (3.0, -2.0, 1.0)),
    ],
)
def test_dipole_preserves_complex_moments_and_arbitrary_positions(dipole_type, moment, position):
    definition = DipoleFieldDefinition(dipole_type, moment, position)
    assert definition.dipole_type == dipole_type
    assert definition.effective_moment == moment
    assert all(isinstance(component, complex) for component in definition.effective_moment)
    assert definition.source_position == position


def test_dipole_is_immutable_even_when_constructed_from_lists():
    moment = [1.0, 1j, 0.0]
    position = [0.1, -0.2, 0.3]
    definition = DipoleFieldDefinition("electric", moment, position)
    moment[0] = 5.0
    position[0] = 5.0
    assert definition.effective_moment == (1.0 + 0j, 1j, 0j)
    assert definition.source_position == (0.1, -0.2, 0.3)
    with pytest.raises(FrozenInstanceError):
        definition.effective_moment = (2.0 + 0j, 0j, 0j)
    with pytest.raises(FrozenInstanceError):
        definition.source_position = (0.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        definition.dipole_type = "magnetic"


@pytest.mark.parametrize("dipole_type", ["invalid", "Electric", "", None, True])
def test_dipole_rejects_invalid_types(dipole_type):
    with pytest.raises(ValueError, match="dipole_type"):
        DipoleFieldDefinition(dipole_type, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "moment, position, message",
    [
        ((1.0, 0.0), (0.0, 0.0, 0.0), "effective_moment.*three"),
        ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "effective_moment.*three"),
        (None, (0.0, 0.0, 0.0), "effective_moment.*three"),
        ((True, 0.0, 0.0), (0.0, 0.0, 0.0), "effective_moment.*finite"),
        (("1", 0.0, 0.0), (0.0, 0.0, 0.0), "effective_moment.*finite"),
        ((float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0), "effective_moment.*finite"),
        ((float("inf"), 0.0, 0.0), (0.0, 0.0, 0.0), "effective_moment.*finite"),
        ((complex(1.0, float("nan")), 0.0, 0.0), (0.0, 0.0, 0.0), "effective_moment.*finite"),
        ((complex(1.0, float("inf")), 0.0, 0.0), (0.0, 0.0, 0.0), "effective_moment.*finite"),
        ((1.0, 0.0, 0.0), (0.0, 0.0), "source_position.*three"),
        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), "source_position.*three"),
        ((1.0, 0.0, 0.0), "abc", "source_position.*three"),
        ((1.0, 0.0, 0.0), (True, 0.0, 0.0), "source_position.*finite"),
        ((1.0, 0.0, 0.0), ("1", 0.0, 0.0), "source_position.*finite"),
        ((1.0, 0.0, 0.0), (1.0 + 0j, 0.0, 0.0), "source_position.*finite"),
        ((1.0, 0.0, 0.0), (float("nan"), 0.0, 0.0), "source_position.*finite"),
        ((1.0, 0.0, 0.0), (float("inf"), 0.0, 0.0), "source_position.*finite"),
    ],
)
def test_dipole_rejects_invalid_vectors(moment, position, message):
    with pytest.raises(ValueError, match=message):
        DipoleFieldDefinition("electric", moment, position)


@pytest.fixture
def workflow_configuration():
    return WorkflowConfiguration(
        fields={
            "plane_x": PlaneWaveFieldDefinition((1.0, 1j, 0.0), (0.0, 0.0, 1.0)),
            "dipole_z": DipoleFieldDefinition("electric", (0.0, 0.0, 1.0 + 2j), (0.0, 0.0, 0.0)),
        },
        transformations=(
            GSTCTransformationConfiguration(
                "state_1",
                GSTCFieldsConfiguration(FieldReference("dipole_z"), FieldReference("plane_x")),
            ),
        ),
        simulation=SimulationConfiguration(
            (
                SimulationCaseConfiguration(
                    "dipole_only",
                    (SimulationExcitationConfiguration(FieldReference("dipole_z"), "interior"),),
                ),
            )
        ),
    )


def test_configuration_catalog_is_read_only_and_detached_from_the_original_mapping(
    workflow_configuration,
):
    catalog = dict(workflow_configuration.fields)
    configuration = replace(workflow_configuration, fields=catalog)
    catalog.clear()
    assert configuration.fields == workflow_configuration.fields
    assert configuration.fields["plane_x"].electric_amplitude == (1.0 + 0j, 1j, 0j)
    assert configuration.fields["dipole_z"].effective_moment == (0j, 0j, 1.0 + 2j)

    with pytest.raises(TypeError):
        configuration.fields["new_field"] = workflow_configuration.fields["plane_x"]
    with pytest.raises(TypeError):
        del configuration.fields["plane_x"]
    with pytest.raises(FrozenInstanceError):
        configuration.fields = {}


@pytest.mark.parametrize("method", ["reciprocal_complex", "reciprocal_lossless"])
@pytest.mark.parametrize("structure", ["general", "diagonal"])
def test_configuration_accepts_supported_extraction_settings(
    workflow_configuration, method, structure
):
    configuration = replace(
        workflow_configuration, extraction_method=method, tensor_structure=structure
    )
    assert configuration.extraction_method == method
    assert configuration.tensor_structure == structure


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"case_id": "../escape"}, "case_id"),
        ({"case_id": ""}, "case_id"),
        ({"case_id": True}, "case_id"),
        ({"extraction_method": "unrestricted"}, "extraction_method"),
        ({"extraction_method": None}, "extraction_method"),
        ({"tensor_structure": "isotropic"}, "tensor_structure"),
        ({"tensor_structure": None}, "tensor_structure"),
        ({"case": {}}, "case must be a SphereCase"),
        ({"case": None}, "case must be a SphereCase"),
        ({"sampling": {}}, "sampling must be a SamplingConfiguration"),
        ({"sampling": None}, "sampling must be a SamplingConfiguration"),
        ({"sie": {}}, "sie must be a SIEConfiguration"),
        ({"sie": None}, "sie must be a SIEConfiguration"),
        ({"fields": []}, "fields must be a mapping"),
        ({"fields": "plane_x"}, "fields must be a mapping"),
        ({"fields": None}, "fields must be a mapping"),
    ],
)
def test_configuration_rejects_invalid_common_settings(workflow_configuration, settings, message):
    with pytest.raises(ValueError, match=message):
        replace(workflow_configuration, **settings)


@pytest.mark.parametrize("field_id", ["", "Plane_X", "../plane_x", "two words", 1, True])
def test_configuration_rejects_invalid_catalog_identifiers(workflow_configuration, field_id):
    with pytest.raises(ValueError, match="fields identifier"):
        replace(workflow_configuration, fields={field_id: workflow_configuration.fields["plane_x"]})


@pytest.mark.parametrize("definition", [None, 1, "plane_wave", {"type": "plane_wave"}])
def test_configuration_rejects_unparsed_or_invalid_catalog_definitions(
    workflow_configuration, definition
):
    with pytest.raises(ValueError, match="fields.plane_x.*definition"):
        replace(workflow_configuration, fields={"plane_x": definition})


@pytest.mark.parametrize("transformations", [(), [], None, (None,)])
def test_configuration_rejects_invalid_transformation_collections(
    workflow_configuration, transformations
):
    with pytest.raises(ValueError, match="transformations"):
        replace(workflow_configuration, transformations=transformations)


def test_configuration_rejects_duplicate_transformation_ids(workflow_configuration):
    with pytest.raises(ValueError, match="transformation IDs must be unique"):
        replace(workflow_configuration, transformations=workflow_configuration.transformations * 2)


@pytest.mark.parametrize("simulation", [None, {}])
def test_configuration_rejects_invalid_simulation_objects(workflow_configuration, simulation):
    with pytest.raises(ValueError, match="simulation must be a SimulationConfiguration"):
        replace(workflow_configuration, simulation=simulation)


@pytest.mark.parametrize("region", ["interior", "exterior"])
def test_configuration_rejects_unknown_gstc_references_with_state_and_region(
    workflow_configuration, region
):
    state = workflow_configuration.transformations[0]
    gstc = replace(state.gstc, **{region: FieldReference("missing")})
    with pytest.raises(ValueError, match=f"state_1.*gstc.{region}.*unknown field 'missing'"):
        replace(workflow_configuration, transformations=(replace(state, gstc=gstc),))


@pytest.mark.parametrize("region", ["interior", "exterior"])
def test_configuration_rejects_unknown_incident_references_with_case_and_region(
    workflow_configuration, region
):
    simulation = SimulationConfiguration(
        (
            SimulationCaseConfiguration(
                "unknown_run",
                (SimulationExcitationConfiguration(FieldReference("missing"), region),),
            ),
        )
    )
    with pytest.raises(ValueError, match=f"unknown_run.*{region}.*unknown.*'missing'"):
        replace(workflow_configuration, simulation=simulation)


@pytest.mark.parametrize(
    "interior, exterior",
    [
        (None, FieldReference("plane_x")),
        (FieldReference("dipole_z"), None),
        (None, None),
    ],
)
def test_configuration_accepts_explicit_zero_gstc_fields(
    workflow_configuration, interior, exterior
):
    gstc = GSTCFieldsConfiguration(interior, exterior)
    state = replace(workflow_configuration.transformations[0], gstc=gstc)
    configuration = replace(workflow_configuration, transformations=(state,))
    assert configuration.transformations[0].gstc == gstc


def test_configuration_simulation_can_reference_a_field_not_used_by_gstc_states(
    workflow_configuration,
):
    catalog = dict(workflow_configuration.fields)
    catalog["plane_y"] = PlaneWaveFieldDefinition((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    simulation = SimulationConfiguration(
        (
            SimulationCaseConfiguration(
                "independent_incidence",
                (SimulationExcitationConfiguration(FieldReference("plane_y"), "exterior"),),
            ),
        )
    )
    configuration = replace(workflow_configuration, fields=catalog, simulation=simulation)
    assert configuration.transformations == workflow_configuration.transformations
    assert configuration.simulation == simulation


def test_configuration_can_reuse_one_field_in_both_regions(workflow_configuration):
    reference = FieldReference("dipole_z")
    state = GSTCTransformationConfiguration(
        "shared_field", GSTCFieldsConfiguration(reference, reference)
    )
    simulation = SimulationConfiguration(
        (
            SimulationCaseConfiguration(
                "both_regions",
                (
                    SimulationExcitationConfiguration(reference, "interior"),
                    SimulationExcitationConfiguration(reference, "exterior"),
                ),
            ),
        )
    )
    configuration = replace(workflow_configuration, transformations=(state,), simulation=simulation)
    assert configuration.transformations == (state,)
    assert configuration.simulation == simulation


@pytest.mark.parametrize("count", [0, 3])
def test_simulation_case_requires_one_or_two_excitations(workflow_configuration, count):
    excitation = workflow_configuration.simulation.cases[0].excitations[0]
    with pytest.raises(ValueError, match="one or two excitations"):
        SimulationCaseConfiguration("invalid_count", (excitation,) * count)


def test_simulation_case_rejects_duplicate_regions(workflow_configuration):
    excitation = workflow_configuration.simulation.cases[0].excitations[0]
    with pytest.raises(ValueError, match="only one excitation per region"):
        SimulationCaseConfiguration("duplicate_region", (excitation, excitation))


def test_simulation_configuration_rejects_duplicate_case_ids(workflow_configuration):
    case = workflow_configuration.simulation.cases[0]
    with pytest.raises(ValueError, match="simulation case IDs must be unique"):
        SimulationConfiguration((case, case))


@pytest.mark.parametrize("dipole_type", ["electric", "magnetic"])
@pytest.mark.parametrize("zero_region", ["interior", "exterior"])
def test_configuration_json_round_trip_preserves_complex_fields_and_all_settings(
    tmp_path, workflow_configuration, dipole_type, zero_region
):
    catalog = dict(workflow_configuration.fields)
    catalog["dipole_z"] = replace(catalog["dipole_z"], dipole_type=dipole_type)
    second_state = replace(
        workflow_configuration.transformations[0],
        id="state_2",
        gstc=replace(workflow_configuration.transformations[0].gstc, **{zero_region: None}),
    )
    both_regions = SimulationCaseConfiguration(
        "combined_incidence",
        (
            SimulationExcitationConfiguration(FieldReference("dipole_z"), "interior"),
            SimulationExcitationConfiguration(FieldReference("plane_x"), "exterior"),
        ),
    )
    configuration = replace(
        workflow_configuration,
        case_id="round_trip",
        fields=catalog,
        transformations=(*workflow_configuration.transformations, second_state),
        simulation=SimulationConfiguration(
            (*workflow_configuration.simulation.cases, both_regions)
        ),
        case=SphereCase(
            frequency_hz=2e8,
            radius_m=2.0,
            relative_permittivity_interior=2.5,
            relative_permeability_interior=1.3,
            relative_permittivity_exterior=1.7,
            relative_permeability_exterior=1.1,
        ),
        extraction_method="reciprocal_lossless",
        tensor_structure="diagonal",
        sampling=replace(workflow_configuration.sampling, radial_phi_degrees=30.0),
        sie=replace(
            workflow_configuration.sie,
            quadrature_rule=13,
            compute_residual=False,
            evaluate_fields=False,
            mesh=replace(
                workflow_configuration.sie.mesh,
                path="meshes/round_trip.obj",
                points_per_wavelength=9.0,
            ),
        ),
    )
    raw = configuration.to_dict()
    assert raw["fields"]["plane_x"]["electric_amplitude"] == {
        "real": [1.0, 0.0, 0.0],
        "imag": [0.0, 1.0, 0.0],
    }
    assert raw["fields"]["dipole_z"]["effective_moment"] == {
        "real": [0.0, 0.0, 1.0],
        "imag": [0.0, 0.0, 2.0],
    }
    assert raw["transformations"][1]["gstc"][zero_region] is None
    path = tmp_path / "configuration.json"
    write_json(path, raw)
    loaded = load_workflow_configuration(path)
    assert loaded == configuration
    assert loaded.to_dict() == raw
    assert json.loads(json.dumps(loaded.to_dict(), allow_nan=False)) == raw
    with pytest.raises(TypeError):
        loaded.fields["new"] = loaded.fields["plane_x"]


def test_configuration_common_settings_can_be_omitted_but_field_contract_is_explicit(
    tmp_path, workflow_configuration
):
    raw = workflow_configuration.to_dict()
    raw = {key: raw[key] for key in ("schema_version", "fields", "transformations", "simulation")}
    path = tmp_path / "minimal.json"
    write_json(path, raw)
    assert load_workflow_configuration(path) == workflow_configuration


@pytest.mark.parametrize(
    "path, value, remove, message",
    [
        (("schema_version",), 1, False, "schema_version"),
        (("schema_version",), True, False, "schema_version"),
        (("schema_version",), 2.0, False, "schema_version"),
        (("schema_version",), None, True, "missing.*schema_version"),
        (("application",), "omega", False, "different application"),
        (("unused",), 1, False, "unknown configuration"),
        (("fields",), None, True, "missing.*fields"),
        (("fields",), [], False, "fields.*JSON object"),
        (("fields", "plane_x"), None, False, "plane_x.*JSON object"),
        (("fields", "plane_x", "type"), "gaussian_beam", False, "plane_x.type"),
        (("fields", "plane_x", "type"), None, True, "plane_x.type"),
        (("fields", "plane_x", "region"), "interior", False, "unknown.*region"),
        (("fields", "plane_x", "scale"), 2, False, "unknown.*scale"),
        (("fields", "plane_x", "electric_amplitude"), [1, 0, 0], False, "JSON object"),
        (("fields", "plane_x", "electric_amplitude", "imag"), None, True, "missing.*imag"),
        (("fields", "plane_x", "electric_amplitude", "real"), [1, 0], False, "three"),
        (("fields", "plane_x", "electric_amplitude", "imag"), [0, True, 0], False, "finite"),
        (
            ("fields", "plane_x", "electric_amplitude", "imag"),
            [0, float("nan"), 0],
            False,
            "finite",
        ),
        (("fields", "plane_x", "electric_amplitude", "unused"), 1, False, "unknown"),
        (("fields", "plane_x", "propagation_direction"), [0, 0, 0], False, "nonzero"),
        (("fields", "plane_x", "electric_amplitude", "real"), [1, 0, 1], False, "transverse"),
        (("fields", "dipole_z", "dipole_type"), "invalid", False, "dipole_type"),
        (("fields", "dipole_z", "source_position"), [0, "0", 0], False, "finite"),
        (("fields", "dipole_z", "effective_moment", "imag"), [0, 0, float("inf")], False, "finite"),
        (("transformations",), None, True, "missing.*transformations"),
        (("transformations",), {}, False, "JSON array"),
        (("transformations",), [], False, "nonempty"),
        (("transformations", 0, "id"), None, True, "missing.*id"),
        (("transformations", 0, "gstc"), None, False, "JSON object"),
        (("transformations", 0, "gstc", "interior"), None, True, "missing.*interior"),
        (("transformations", 0, "gstc", "exterior"), None, True, "missing.*exterior"),
        (("transformations", 0, "gstc", "interior"), "missing", False, "unknown field"),
        (("transformations", 0, "gstc", "interior"), {"field": "dipole_z"}, False, "interior"),
        (("simulation",), None, True, "missing.*simulation"),
        (("simulation", "cases"), None, True, "missing.*cases"),
        (("simulation", "cases"), {}, False, "JSON array"),
        (("simulation", "cases", 0, "excitations"), None, True, "missing.*excitations"),
        (("simulation", "cases", 0, "excitations"), [], False, "one or two"),
        (("simulation", "cases", 0, "excitations", 0, "field"), None, False, "field"),
        (("simulation", "cases", 0, "excitations", 0, "field"), "missing", False, "unknown"),
        (("simulation", "cases", 0, "excitations", 0, "region"), None, True, "missing.*region"),
        (("simulation", "cases", 0, "excitations", 0, "region"), "both", False, "region"),
        (("physical", "radius_m"), 0, False, "radius_m"),
        (("sampling", "radial_points"), True, False, "integer"),
        (("sie", "mesh", "points_per_wavelength"), 0, False, "points_per_wavelength"),
    ],
)
def test_configuration_json_rejects_invalid_or_incomplete_contract(
    tmp_path, workflow_configuration, path, value, remove, message
):
    raw = workflow_configuration.to_dict()
    parent = raw
    for key in path[:-1]:
        parent = parent[key]
    if remove:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value
    file_path = tmp_path / "invalid.json"
    file_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_workflow_configuration(file_path)
    assert list(tmp_path.iterdir()) == [file_path]


@pytest.mark.parametrize("section", ["transformations", "cases", "excitations"])
def test_configuration_json_rejects_duplicate_ids_and_regions(
    tmp_path, workflow_configuration, section
):
    raw = workflow_configuration.to_dict()
    if section == "transformations":
        items = raw["transformations"]
        message = "transformation IDs"
    elif section == "cases":
        items = raw["simulation"]["cases"]
        message = "case IDs"
    else:
        items = raw["simulation"]["cases"][0]["excitations"]
        message = "one excitation per region"
    items.append(items[0])
    path = tmp_path / "duplicates.json"
    write_json(path, raw)
    with pytest.raises(ValueError, match=message):
        load_workflow_configuration(path)


def test_configuration_json_rejects_duplicate_object_keys(tmp_path, workflow_configuration):
    encoded = json.dumps(workflow_configuration.to_dict())
    path = tmp_path / "duplicate_key.json"
    path.write_text(
        encoded.replace('"schema_version": 2', '"schema_version": 2, "schema_version": 2')
    )
    with pytest.raises(ValueError, match="duplicate JSON field: schema_version"):
        load_workflow_configuration(path)


def test_configuration_json_rejects_nonobject_root(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="configuration must be a JSON object"):
        load_workflow_configuration(path)


def test_legacy_schema_is_not_supported(tmp_path, workflow_configuration):
    path = tmp_path / "legacy.json"
    raw = workflow_configuration.to_dict()
    raw["schema_version"] = 1
    write_json(path, raw)
    with pytest.raises(ValueError, match="schema_version must be 2"):
        load_workflow_configuration(path)
