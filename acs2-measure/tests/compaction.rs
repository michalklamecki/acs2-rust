use std::fs;
use std::process::Command;

use acs2_agents::acs2vcp::Preset as VcpPreset;
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_measure::compaction_spec::{CompactionSpecification, ShadowSpecification};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{
    run_protocol_shadowed_with_sink, AgentKind, CompactingFamily, RunInputs, RunMetadata, RunSettings,
};
use serde_json::{json, Value};

const TARGETS: [u64; 2] = [60, 150];

const COMPACTING_NAMES: [&str; 8] = [
    "acs2_compacting",
    "acs2er_compacting",
    "acs2er_episodic_compacting",
    "acs2her_compacting",
    "acs2vcp_k2_compacting",
    "acs2vcp_k4_compacting",
    "acs2vcp_k2_uniform_compacting",
    "acs2vcp_k4_uniform_compacting",
];

fn specification(mode: &str, family: CompactingFamily, trigger: Value) -> &'static CompactionSpecification {
    let evidence_free = family == CompactingFamily::Acs2;
    let vcp = matches!(family, CompactingFamily::Acs2Vcp(_));
    let mut value = json!({
        "mode": mode,
        "clock": if vcp { json!("per_head") } else { Value::Null },
        "theta_r": 0.9,
        "operators": [{"kind": "mature_subsumption", "guarded": true}],
        "trigger": trigger,
        "strength": {"value_tolerance": 0.0, "maturity_experience": 2, "idle_audits": 1},
        "strength_bounds": null,
        "feedback": null,
        "phase_modulation": null,
        "audit": if evidence_free {
            json!({"policy": "skip"})
        } else {
            json!({"policy": "enforce", "max_coverage_loss": 0.0, "max_no_change_coverage_loss": 0.0,
                "max_value_change": 0.0, "max_greedy_change_fraction": 0.0})
        },
        "evidence_sample": if evidence_free { 0 } else { 40 },
        "census": true,
        "acceptance_limits": null,
        "disagreement_probe": null,
        "max_records_per_row": 100
    });
    if !vcp {
        value.as_object_mut().unwrap().remove("clock");
    }
    CompactionSpecification::from_json(&value.to_string())
        .unwrap()
        .leak()
}

fn periodic(every: u64) -> Value {
    json!({"kind": "periodic", "every_interactions": every})
}

fn shadow_specification(evidence: bool) -> &'static ShadowSpecification {
    let entry = |label: &str, operators: Value, idle_audits: u32| {
        json!({"label": label, "operators": operators,
            "strength": {"value_tolerance": 0.0, "maturity_experience": 2, "idle_audits": idle_audits}})
    };
    let mut entries = vec![
        entry("mature", json!([{"kind": "mature_subsumption", "guarded": true}]), 1),
        entry("cracs", json!([{"kind": "cracs_subsumption", "guarded": false}]), 1),
        entry(
            "thresholds",
            json!([{"kind": "dynamic_thresholds", "size_threshold": 1, "experience_threshold": 1, "quality_threshold": 0.9}]),
            1,
        ),
    ];
    if evidence {
        entries.push(entry("idle", json!([{"kind": "idle_pruning", "min_experience": 1}]), 2));
        entries.push(entry("capacity", json!([{"kind": "capacity", "capacity": 3}]), 1));
    }
    let value = json!({
        "theta_r": 0.9,
        "evidence_sample": if evidence { 40 } else { 0 },
        "audit": if evidence {
            json!({"max_coverage_loss": 0.0, "max_no_change_coverage_loss": 0.0,
                "max_value_change": 0.0, "max_greedy_change_fraction": 0.0})
        } else {
            Value::Null
        },
        "census": true,
        "entries": entries
    });
    ShadowSpecification::from_json(&value.to_string()).unwrap().leak()
}

struct Outcome {
    rows: Vec<Value>,
    population: String,
    state: String,
}

fn run(kind: AgentKind, shadow: Option<&'static ShadowSpecification>) -> Outcome {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let mut rows = Vec::new();
    let (population, state) = run_protocol_shadowed_with_sink::<_, _, 8, 2, 10>(
        &distributions.protocol(),
        kind,
        shadow,
        RunInputs {
            seed: 42,
            targets: &TARGETS,
            metadata: &RunMetadata::test("test"),
            settings: RunSettings {
                evaluate_points: true,
                capture_final_state: true,
            },
        },
        &mut |row| rows.push(row),
    )
    .expect("state capture enabled");
    Outcome {
        rows,
        population,
        state,
    }
}

fn learning_fields(row: &Value) -> Value {
    let mut row = row.clone();
    let object = row.as_object_mut().unwrap();
    for key in [
        "agent",
        "agent_parameters",
        "compaction",
        "compaction_shadow",
        "wall_seconds_train",
        "wall_seconds_eval",
        "wall_seconds_total",
    ] {
        object.remove(key);
    }
    if let Some(ensemble) = object.get_mut("ensemble").and_then(Value::as_object_mut) {
        ensemble.remove("measurement_diagnostics_inline_bytes");
        ensemble.remove("agent_and_measurement_known_bytes_lower_bound");
    }
    row
}

fn plain_counterpart(name: &str) -> AgentKind {
    CompactingFamily::from_run_name(name).unwrap().plain()
}

fn compacting(name: &str, mode: &str, trigger: Value) -> AgentKind {
    let family = CompactingFamily::from_run_name(name).unwrap();
    AgentKind::compacting(family, specification(mode, family, trigger)).unwrap()
}

#[test]
fn compacting_run_names_are_distinct_content_names_that_need_a_specification() {
    let mut seen = Vec::new();
    for name in COMPACTING_NAMES {
        let family = CompactingFamily::from_run_name(name).unwrap();
        assert_eq!(family.name(), name);
        assert!(!seen.contains(&name));
        seen.push(name);
        let error = AgentKind::from_run_name(name, 0.5).unwrap_err();
        assert!(error.contains("--compaction"), "{error}");
        let kind = compacting(name, "off", periodic(50));
        assert_eq!(kind.name(), name);
        assert_eq!(acs2_measure::agent_names::canonical_run_name(name), name);
    }
    assert!(CompactingFamily::from_run_name("acs2vcp_k2").is_none());
}

#[test]
fn a_specification_is_validated_against_the_family_when_the_kind_is_built() {
    let evidence_family = CompactingFamily::Acs2Er;
    let spec = specification("apply", evidence_family, periodic(50));
    let error = AgentKind::compacting(CompactingFamily::Acs2, spec).unwrap_err();
    assert!(error.contains("no experience memory"), "{error}");
    let error = AgentKind::compacting(
        CompactingFamily::Acs2Vcp(VcpPreset::UniformTwo),
        spec,
    )
    .unwrap_err();
    assert!(error.contains("clock"), "{error}");
}

#[test]
fn mode_off_reproduces_the_learning_outcome_of_every_plain_counterpart() {
    for name in COMPACTING_NAMES {
        let plain = run(plain_counterpart(name), None);
        let off = run(compacting(name, "off", periodic(50)), None);
        assert_eq!(plain.rows.len(), off.rows.len());
        for (plain_row, off_row) in plain.rows.iter().zip(&off.rows) {
            assert_eq!(learning_fields(plain_row), learning_fields(off_row), "{name}");
            assert_eq!(off_row["agent"], name);
            assert!(plain_row.get("compaction").is_none());
            assert_eq!(off_row["compaction"]["reductions"], 0);
            assert_eq!(off_row["agent_parameters"]["compaction"]["mode"], "off");
            assert_eq!(off_row["agent_parameters"]["compaction_stream"], 11);
        }
        assert_eq!(plain.population, off.population, "{name}");
        if !name.starts_with("acs2vcp") {
            assert_eq!(plain.state, off.state, "{name}");
        }
    }
}

#[test]
fn off_rows_report_the_population_size_area_without_any_reduction() {
    let outcome = run(compacting("acs2her_compacting", "off", periodic(50)), None);
    let last = outcome.rows.last().unwrap();
    let area = &last["compaction"]["population_size_area"];
    assert_eq!(area["interactions"], last["actual_steps"]);
    assert!(area["area"].as_u64().unwrap() > 0);
}

#[test]
fn shadow_at_checkpoints_leaves_every_learning_field_of_every_agent_unchanged() {
    let evidence = shadow_specification(true);
    let evidence_free = shadow_specification(false);
    let kinds = [
        (AgentKind::Acs2, evidence_free),
        (AgentKind::Acs2Er, evidence),
        (AgentKind::Acs2ErOneUpdate, evidence_free),
        (AgentKind::Acs2ErEpisodic, evidence),
        (AgentKind::Acs2Her, evidence),
        (AgentKind::Acs2Vcp(VcpPreset::PrioritizedTwo { temperature: 4 }), evidence),
        (AgentKind::Acs2Vcp(VcpPreset::UniformFour), evidence),
    ];
    for (kind, shadow) in kinds {
        let baseline = run(kind, None);
        let shadowed = run(kind, Some(shadow));
        assert_eq!(baseline.population, shadowed.population, "{}", kind.name());
        assert_eq!(baseline.state, shadowed.state, "{}", kind.name());
        let heads = baseline.rows[0]["ensemble"]["heads"]
            .as_array()
            .map_or(1, Vec::len);
        for (before, after) in baseline.rows.iter().zip(&shadowed.rows) {
            assert_eq!(
                {
                    let mut row = before.clone();
                    row.as_object_mut().unwrap().remove("wall_seconds_train");
                    row.as_object_mut().unwrap().remove("wall_seconds_eval");
                    row.as_object_mut().unwrap().remove("wall_seconds_total");
                    row
                },
                {
                    let mut row = after.clone();
                    let object = row.as_object_mut().unwrap();
                    object.remove("compaction_shadow");
                    object.remove("wall_seconds_train");
                    object.remove("wall_seconds_eval");
                    object.remove("wall_seconds_total");
                    row
                },
                "{}",
                kind.name()
            );
            assert!(before.get("compaction_shadow").is_none());
            assert!(before.get("compaction").is_none());
            let report = &after["compaction_shadow"];
            assert_eq!(report["interactions"], after["actual_steps"]);
            let entries = report["entries"].as_array().unwrap();
            assert_eq!(entries.len(), shadow.entries.len());
            for entry in entries {
                let per_head = entry["heads"].as_array().unwrap();
                assert_eq!(per_head.len(), heads);
                for record in per_head {
                    assert_eq!(record["mode"], "shadow");
                    assert_eq!(record["applied"], false);
                    assert_eq!(record["size_after"], record["size_before"]);
                    assert!(record["census_before"].is_object());
                }
            }
        }
    }
}

#[test]
fn shadow_records_the_audit_of_the_candidate_when_evidence_is_available() {
    let outcome = run(AgentKind::Acs2Her, Some(shadow_specification(true)));
    let last = outcome.rows.last().unwrap();
    let entries = last["compaction_shadow"]["entries"].as_array().unwrap();
    assert!(last["compaction_shadow"]["evidence_size"].as_u64().unwrap() > 0);
    for entry in entries {
        let record = &entry["heads"][0];
        assert_eq!(record["size_before"], last["population_classifiers"]);
        assert!(record["audit"]["coverage_before"]["all"]["transitions"].as_u64().unwrap() > 0);
        assert!(record["audit"]["verdict"].is_object());
        assert!(record["work"]["operator"].is_u64());
    }
}

#[test]
fn shadow_without_evidence_is_refused_for_an_agent_that_keeps_no_memory() {
    assert!(!AgentKind::Acs2.keeps_experience_memory());
    assert!(!AgentKind::Acs2ErOneUpdate.keeps_experience_memory());
    assert!(AgentKind::Acs2Er.keeps_experience_memory());
    let refused = std::panic::catch_unwind(|| run(AgentKind::Acs2, Some(shadow_specification(true))));
    assert!(refused.is_err());
}

#[test]
fn apply_mode_keeps_the_formation_identity_and_reports_its_work_separately() {
    for name in [
        "acs2er_episodic_compacting",
        "acs2her_compacting",
        "acs2vcp_k2_compacting",
        "acs2vcp_k4_uniform_compacting",
    ] {
        let outcome = run(compacting(name, "apply", periodic(20)), None);
        let mut reductions = 0;
        for row in &outcome.rows {
            let steps = row["actual_steps"].as_u64().unwrap();
            let counts = &row["learning_update_counts"];
            let full = counts["full"].as_u64().unwrap();
            let values = counts["values_only"].as_u64().unwrap();
            let queries = row["replay_diagnostics"]["priority"]["value_queries"]
                .as_u64()
                .unwrap_or(0);
            assert_eq!(
                row["match_formations_train"],
                steps + 3 * full + 2 * values + queries,
                "{name}"
            );
            let compaction = &row["compaction"];
            reductions = compaction["reductions"].as_u64().unwrap();
            assert_eq!(
                compaction["accepted"].as_u64().unwrap() + compaction["rejected"].as_u64().unwrap(),
                reductions
            );
            assert!(compaction["work"]["operator"].is_u64());
            assert!(compaction["training_cost"]["classifier_tests"].is_u64());
            assert_eq!(row["agent"], name);
        }
        assert!(reductions > 0, "{name}");
    }
}

#[test]
fn rows_carry_the_reductions_since_the_previous_row_and_the_running_totals() {
    for name in ["acs2her_compacting", "acs2vcp_k2_compacting", "acs2er_compacting", "acs2_compacting"] {
        let outcome = run(compacting(name, "apply", periodic(20)), None);
        let mut total = 0;
        let mut previous = 0;
        for row in &outcome.rows {
            let compaction = &row["compaction"];
            let reductions = compaction["reductions"].as_u64().unwrap();
            let fresh = compaction["records_new"].as_u64().unwrap();
            assert_eq!(fresh, reductions - previous, "{name}");
            assert_eq!(compaction["records_omitted"], 0);
            assert_eq!(compaction["records"].as_array().unwrap().len() as u64, fresh);
            assert_eq!(compaction["records_scope"], "reductions_since_previous_row");
            previous = reductions;
            total += fresh;
        }
        assert_eq!(total, previous);
        assert!(total > 0, "{name}");
    }
}

#[test]
fn the_record_limit_bounds_a_row_and_counts_what_it_omitted() {
    let family = CompactingFamily::Acs2Her;
    let mut value = specification("apply", family, periodic(10)).echo();
    value["max_records_per_row"] = json!(2);
    let limited = CompactionSpecification::from_json(&value.to_string())
        .unwrap()
        .leak();
    let outcome = run(AgentKind::compacting(family, limited).unwrap(), None);
    let first = &outcome.rows[0]["compaction"];
    assert!(first["records_new"].as_u64().unwrap() > 2);
    assert_eq!(first["records"].as_array().unwrap().len(), 2);
    assert_eq!(
        first["records_omitted"].as_u64().unwrap(),
        first["records_new"].as_u64().unwrap() - 2
    );
}

#[test]
fn a_record_carries_the_full_reduction_measurement() {
    let family = CompactingFamily::Acs2Her;
    let mut value = specification("apply", family, periodic(20)).echo();
    value["operators"] = json!([{"kind": "dynamic_thresholds", "size_threshold": 1,
        "experience_threshold": 0, "quality_threshold": 0.99}]);
    let configured = CompactionSpecification::from_json(&value.to_string())
        .unwrap()
        .leak();
    let outcome = run(AgentKind::compacting(family, configured).unwrap(), None);
    let records: Vec<&Value> = outcome
        .rows
        .iter()
        .flat_map(|row| row["compaction"]["records"].as_array().unwrap())
        .collect();
    let record = records
        .iter()
        .find(|record| record["audit"].is_object())
        .expect("a reduction with a candidate is audited");
    for key in [
        "head",
        "sequence",
        "mode",
        "interaction",
        "observation",
        "strength",
        "evidence_size",
        "size_before",
        "size_candidate",
        "size_after",
        "removals",
        "value_vetoes",
        "stages",
        "work",
        "audit",
        "check_vetoed",
        "accepted",
        "applied",
        "restoration",
        "census_before",
        "census_candidate",
        "feedback",
    ] {
        assert!(record.get(key).is_some(), "{key}");
    }
    for key in ["subsumed", "idle", "low_quality", "evicted", "total"] {
        assert!(record["removals"].get(key).is_some(), "{key}");
    }
    assert!(record["audit"]["verdict"]["accepted"].is_boolean());
    assert!(record["audit"]["coverage_before"]["no_change"]["transitions"].is_u64());
    assert!(record["size_candidate"].as_u64() < record["size_before"].as_u64());
}

#[test]
fn the_vcp_diagnostics_include_the_acceptance_audit_and_the_disagreement_probe() {
    let family = CompactingFamily::Acs2Vcp(VcpPreset::PrioritizedTwo { temperature: 4 });
    let mut value = specification("apply", family, periodic(20)).echo();
    value["operators"] = json!([{"kind": "dynamic_thresholds", "size_threshold": 1,
        "experience_threshold": 0, "quality_threshold": 0.99}]);
    value["audit"] = json!({"policy": "skip"});
    value["acceptance_limits"] =
        json!({"max_mean_acceptance_shift": 1.0, "max_mean_variance_shift": 100000.0});
    value["disagreement_probe"] = json!({"every_episodes": 4, "sample": 20});
    let configured = CompactionSpecification::from_json(&value.to_string())
        .unwrap()
        .leak();
    let outcome = run(AgentKind::compacting(family, configured).unwrap(), None);
    let vcp = &outcome.rows.last().unwrap()["compaction"]["vcp"];
    assert_eq!(vcp["clock"], "per_head");
    assert!(vcp["acceptance_audits_total"].as_u64().unwrap() > 0);
    assert!(vcp["disagreement_probes"].as_array().is_some());
    let probes: usize = outcome
        .rows
        .iter()
        .map(|row| row["compaction"]["vcp"]["disagreement_probes"].as_array().unwrap().len())
        .sum();
    assert!(probes > 0);
    assert_eq!(outcome.rows.last().unwrap()["compaction"]["heads"].as_array().unwrap().len(), 2);
}

#[test]
fn a_synchronous_clock_reports_one_policy_for_all_heads() {
    let family = CompactingFamily::Acs2Vcp(VcpPreset::UniformTwo);
    let mut value = specification("apply", family, periodic(20)).echo();
    value["clock"] = json!("synchronous");
    let synchronous = CompactionSpecification::from_json(&value.to_string())
        .unwrap()
        .leak();
    let outcome = run(AgentKind::compacting(family, synchronous).unwrap(), None);
    let compaction = &outcome.rows.last().unwrap()["compaction"];
    assert!(compaction["synchronous_policy"]["fires"].as_u64().unwrap() > 0);
    assert_eq!(compaction["vcp"]["clock"], "synchronous");
}

#[test]
fn compacting_runs_are_deterministic_and_do_not_perturb_the_wall_clock_free_fields() {
    for name in ["acs2her_compacting", "acs2vcp_k2_compacting"] {
        let first = run(compacting(name, "apply", periodic(20)), None);
        let second = run(compacting(name, "apply", periodic(20)), None);
        assert_eq!(first.population, second.population);
        assert_eq!(first.state, second.state);
        for (a, b) in first.rows.iter().zip(&second.rows) {
            let strip = |row: &Value| {
                let mut row = row.clone();
                for key in ["wall_seconds_train", "wall_seconds_eval", "wall_seconds_total"] {
                    row.as_object_mut().unwrap().remove(key);
                }
                row
            };
            assert_eq!(strip(a), strip(b), "{name}");
        }
    }
}

fn write_fixture(name: &str, value: &Value) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("acs2-compaction-{}-{name}", std::process::id()));
    fs::write(&path, value.to_string()).unwrap();
    path
}

fn cli(agents: &str, extra: &[&std::path::Path], flags: &[&str], out: &std::path::Path) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_acs2-measure"));
    command.args([
        "--task", "maze4", "--cap", "5", "--pool", "2:5,5:5,6:3,6:4", "--encoding", "coordinates",
        "--agents", agents, "--seeds", "42", "--targets", "30", "--out",
    ]);
    command.arg(out);
    for (flag, path) in flags.iter().zip(extra) {
        command.arg(flag).arg(path);
    }
    command.output().expect("runner process")
}

#[test]
fn the_command_line_requires_a_specification_exactly_for_the_compacting_run_names() {
    let out = std::env::temp_dir().join(format!("acs2-compaction-{}-rejected.jsonl", std::process::id()));
    let spec = write_fixture("spec.json", &specification("apply", CompactingFamily::Acs2Her, periodic(20)).echo());
    let missing = cli("acs2her_compacting", &[], &[], &out);
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("needs --compaction"));
    let superfluous = cli("acs2", &[&spec], &["--compaction"], &out);
    assert!(!superfluous.status.success());
    assert!(String::from_utf8_lossy(&superfluous.stderr).contains("applies only to the compacting run names"));
    let mut bad = specification("apply", CompactingFamily::Acs2Her, periodic(20)).echo();
    bad["surprise"] = json!(1);
    let bad_path = write_fixture("bad.json", &bad);
    let rejected = cli("acs2her_compacting", &[&bad_path], &["--compaction"], &out);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("surprise"));
    let mut zero = specification("apply", CompactingFamily::Acs2Her, periodic(20)).echo();
    zero["trigger"] = json!({"kind": "periodic", "every_interactions": 0});
    let zero_path = write_fixture("zero.json", &zero);
    let invalid = cli("acs2her_compacting", &[&zero_path], &["--compaction"], &out);
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("period"));
    for path in [spec, bad_path, zero_path] {
        fs::remove_file(path).unwrap();
    }
    assert!(!out.exists());
}

#[test]
fn the_command_line_writes_rows_with_the_specification_and_the_diagnostics() {
    let out = std::env::temp_dir().join(format!("acs2-compaction-{}-rows.jsonl", std::process::id()));
    let spec_value = specification("apply", CompactingFamily::Acs2Her, periodic(20)).echo();
    let spec = write_fixture("rows-spec.json", &spec_value);
    let shadow = write_fixture("rows-shadow.json", &shadow_specification(true).echo());
    let output = cli(
        "acs2her_compacting,acs2her",
        &[&spec, &shadow],
        &["--compaction", "--compaction-shadow"],
        &out,
    );
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let contents = fs::read_to_string(&out).unwrap();
    fs::remove_file(&out).unwrap();
    for path in [spec, shadow] {
        fs::remove_file(path).unwrap();
    }
    let rows: Vec<Value> = contents.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["agent"], "acs2her_compacting");
    assert_eq!(rows[0]["agent_parameters"]["compaction"], spec_value);
    assert!(rows[0]["compaction"].is_object());
    assert!(rows[0]["compaction_shadow"].is_object());
    assert_eq!(rows[1]["agent"], "acs2her");
    assert!(rows[1].get("compaction").is_none());
    assert!(rows[1].get("agent_parameters").unwrap().get("compaction").is_none());
    assert!(rows[1]["compaction_shadow"].is_object());
}

#[test]
fn the_vcp_temperature_flag_reaches_a_prioritized_compacting_run() {
    let out = std::env::temp_dir().join(format!("acs2-compaction-{}-temperature.jsonl", std::process::id()));
    let family = CompactingFamily::Acs2Vcp(VcpPreset::PrioritizedTwo { temperature: 1 });
    let spec = write_fixture("temperature-spec.json", &specification("off", family, periodic(20)).echo());
    let output = Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
        .args([
            "--task", "maze4", "--cap", "5", "--pool", "2:5,5:5,6:3,6:4", "--encoding", "coordinates",
            "--agents", "acs2vcp_k2_compacting", "--seeds", "42", "--targets", "30",
            "--vcp-temperature", "7", "--compaction",
        ])
        .arg(&spec)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("runner process");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let contents = fs::read_to_string(&out).unwrap();
    fs::remove_file(&out).unwrap();
    fs::remove_file(&spec).unwrap();
    let row: Value = serde_json::from_str(contents.lines().next().unwrap()).unwrap();
    assert_eq!(row["agent_parameters"]["temperature"], 7);
    assert_eq!(row["agent"], "acs2vcp_k2_compacting");
}

#[test]
fn every_example_specification_parses_and_fits_the_families_it_is_meant_for() {
    use acs2_measure::compaction_spec::Family;
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../experiments/rule-compaction/specifications");
    let mut checked = 0;
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        if name.starts_with("shadow_") {
            let specification = ShadowSpecification::from_path(&path).unwrap();
            assert!(specification.entries().unwrap().len() >= 6, "{name}");
        } else {
            let specification = CompactionSpecification::from_path(&path).unwrap();
            for family in [Family::Acs2Er, Family::Episodic, Family::Vcp] {
                specification.validate_for(family).unwrap_or_else(|error| panic!("{name}: {error}"));
            }
            assert_eq!(
                specification.validate_for(Family::Acs2).is_ok(),
                name.ends_with("_without_evidence.json"),
                "{name}"
            );
            assert_eq!(specification.echo(), serde_json::from_str::<Value>(&fs::read_to_string(&path).unwrap()).unwrap(), "{name}");
        }
        checked += 1;
    }
    assert_eq!(checked, 6);
}
