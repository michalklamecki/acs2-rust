use std::fmt;
use std::path::Path;

use acs2_agents::acs2vcp::compacting::{
    AcceptanceLimits, CompactionClock, CompactingConfiguration as VcpCompactingConfiguration,
    DisagreementProbe,
};
use acs2_core::compaction::trigger::{
    CostBenefit, FeedbackParameters, Growth, Never, Once, Periodic, PhaseModulation, StrengthStep,
};
use acs2_core::compaction::{
    fixed_strength_policy, AuditPolicy, AuditThresholds, BaseTrigger, CapacityConfiguration,
    CompactionMode, CompactionParameters, CompactionPolicy, CompactionStrength, DynamicThresholds,
    Operator, StrengthBounds,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecificationError(String);

impl SpecificationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    fn wrap(context: &str, error: impl fmt::Display) -> Self {
        Self(format!("{context}: {error}"))
    }
}

impl fmt::Display for SpecificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for SpecificationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeSpecification {
    Off,
    Shadow,
    Apply,
}

impl ModeSpecification {
    fn core(self) -> CompactionMode {
        match self {
            Self::Off => CompactionMode::Off,
            Self::Shadow => CompactionMode::Shadow,
            Self::Apply => CompactionMode::Apply,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockSpecification {
    PerHead,
    Synchronous,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrengthSpecification {
    pub value_tolerance: f64,
    pub maturity_experience: u32,
    pub idle_audits: u32,
}

impl StrengthSpecification {
    fn core(self) -> CompactionStrength {
        CompactionStrength {
            value_tolerance: self.value_tolerance,
            maturity_experience: self.maturity_experience,
            idle_audits: self.idle_audits,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrengthStepSpecification {
    pub value_tolerance: f64,
    pub maturity_experience: u32,
    pub idle_audits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrengthBoundsSpecification {
    pub gentlest: StrengthSpecification,
    pub boldest: StrengthSpecification,
    pub step: StrengthStepSpecification,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperatorSpecification {
    MatureSubsumption {
        guarded: bool,
    },
    CracsSubsumption {
        guarded: bool,
    },
    IdlePruning {
        min_experience: u32,
    },
    DynamicThresholds {
        size_threshold: usize,
        experience_threshold: u32,
        quality_threshold: f64,
    },
    Capacity {
        capacity: usize,
    },
}

impl OperatorSpecification {
    fn core(self) -> Operator {
        match self {
            Self::MatureSubsumption { guarded } => Operator::MatureSubsumption { guarded },
            Self::CracsSubsumption { guarded } => Operator::CracsSubsumption { guarded },
            Self::IdlePruning { min_experience } => Operator::IdlePruning { min_experience },
            Self::DynamicThresholds {
                size_threshold,
                experience_threshold,
                quality_threshold,
            } => Operator::DynamicThresholds(DynamicThresholds::new(
                size_threshold,
                experience_threshold,
                quality_threshold,
            )),
            Self::Capacity { capacity } => Operator::Capacity(CapacityConfiguration::new(capacity)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TriggerSpecification {
    Never,
    Once {
        at_interaction: u64,
    },
    Periodic {
        every_interactions: u64,
    },
    Growth {
        ratio_permille: u32,
        minimum_reference: usize,
    },
    CostBenefit {
        prior_yield_permille: u32,
        yield_floor_permille: u32,
        margin_permille: u32,
    },
}

impl TriggerSpecification {
    fn core(self) -> Result<BaseTrigger, SpecificationError> {
        let trigger = |error| SpecificationError::wrap("trigger", error);
        Ok(match self {
            Self::Never => Never.into(),
            Self::Once { at_interaction } => Once::new(at_interaction).into(),
            Self::Periodic { every_interactions } => {
                Periodic::new(every_interactions).map_err(trigger)?.into()
            }
            Self::Growth {
                ratio_permille,
                minimum_reference,
            } => Growth::new(ratio_permille, minimum_reference)
                .map_err(trigger)?
                .into(),
            Self::CostBenefit {
                prior_yield_permille,
                yield_floor_permille,
                margin_permille,
            } => CostBenefit::new(prior_yield_permille, yield_floor_permille, margin_permille)
                .map_err(trigger)?
                .into(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedbackSpecification {
    pub low_yield_permille: u32,
    pub high_yield_permille: u32,
    pub restored_bound_permille: u32,
    pub minimum_holdoff: u64,
    pub initial_holdoff: u64,
    pub maximum_holdoff: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseModulationSpecification {
    pub creation_rate_ceiling_per_thousand: Option<u64>,
    pub coverage_floor_permille: Option<u32>,
    pub strength_steps: u32,
    pub holdoff_scale_permille: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdsSpecification {
    pub max_coverage_loss: f64,
    pub max_no_change_coverage_loss: f64,
    pub max_value_change: f64,
    pub max_greedy_change_fraction: f64,
}

impl ThresholdsSpecification {
    fn core(self) -> AuditThresholds {
        AuditThresholds::new(
            self.max_coverage_loss,
            self.max_no_change_coverage_loss,
            self.max_value_change,
            self.max_greedy_change_fraction,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum AuditSpecification {
    Skip,
    Observe(ThresholdsSpecification),
    Enforce(ThresholdsSpecification),
}

impl AuditSpecification {
    fn core(self) -> AuditPolicy {
        match self {
            Self::Skip => AuditPolicy::Skip,
            Self::Observe(thresholds) => AuditPolicy::Observe(thresholds.core()),
            Self::Enforce(thresholds) => AuditPolicy::Enforce(thresholds.core()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptanceLimitsSpecification {
    pub max_mean_acceptance_shift: f64,
    pub max_mean_variance_shift: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisagreementProbeSpecification {
    pub every_episodes: u64,
    pub sample: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Acs2,
    Acs2Er,
    Episodic,
    Vcp,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactionSpecification {
    pub mode: ModeSpecification,
    pub clock: Option<ClockSpecification>,
    pub theta_r: f64,
    pub operators: Vec<OperatorSpecification>,
    pub trigger: TriggerSpecification,
    pub strength: StrengthSpecification,
    pub strength_bounds: Option<StrengthBoundsSpecification>,
    pub feedback: Option<FeedbackSpecification>,
    pub phase_modulation: Option<PhaseModulationSpecification>,
    pub audit: AuditSpecification,
    pub evidence_sample: usize,
    pub census: bool,
    pub acceptance_limits: Option<AcceptanceLimitsSpecification>,
    pub disagreement_probe: Option<DisagreementProbeSpecification>,
    pub max_records_per_row: usize,
}

fn read(path: &Path) -> Result<String, SpecificationError> {
    std::fs::read_to_string(path)
        .map_err(|error| SpecificationError::new(format!("{}: {error}", path.display())))
}

impl CompactionSpecification {
    pub fn from_json(text: &str) -> Result<Self, SpecificationError> {
        let specification: Self = serde_json::from_str(text)
            .map_err(|error| SpecificationError::wrap("compaction specification", error))?;
        specification.parameters()?;
        Ok(specification)
    }

    pub fn from_path(path: &Path) -> Result<Self, SpecificationError> {
        Self::from_json(&read(path)?)
            .map_err(|error| SpecificationError::new(format!("{}: {error}", path.display())))
    }

    pub fn leak(self) -> &'static Self {
        Box::leak(Box::new(self))
    }

    pub fn echo(&self) -> Value {
        serde_json::to_value(self).expect("a specification serializes")
    }

    fn policy(&self) -> Result<CompactionPolicy, SpecificationError> {
        let trigger = self.trigger.core()?;
        let strength = self.strength.core();
        let mut policy = match &self.strength_bounds {
            None => {
                if self.feedback.is_some() {
                    return Err(SpecificationError::new(
                        "feedback adjusts the strength and needs strength_bounds",
                    ));
                }
                fixed_strength_policy(trigger, strength)
                    .map_err(|error| SpecificationError::wrap("strength", error))?
            }
            Some(bounds) => {
                let step = StrengthStep::new(
                    bounds.step.value_tolerance,
                    bounds.step.maturity_experience,
                    bounds.step.idle_audits,
                )
                .map_err(|error| SpecificationError::wrap("strength step", error))?;
                let bounds = StrengthBounds::new(bounds.gentlest.core(), bounds.boldest.core(), step)
                    .map_err(|error| SpecificationError::wrap("strength bounds", error))?;
                CompactionPolicy::new(trigger, bounds, strength)
            }
        };
        if let Some(feedback) = self.feedback {
            policy = policy.with_feedback(
                FeedbackParameters::new(
                    feedback.low_yield_permille,
                    feedback.high_yield_permille,
                    feedback.restored_bound_permille,
                    feedback.minimum_holdoff,
                    feedback.initial_holdoff,
                    feedback.maximum_holdoff,
                )
                .map_err(|error| SpecificationError::wrap("feedback", error))?,
            );
        }
        if let Some(phase) = self.phase_modulation {
            policy = policy.with_phase_modulation(
                PhaseModulation::new(
                    phase.creation_rate_ceiling_per_thousand,
                    phase.coverage_floor_permille,
                    phase.strength_steps,
                    phase.holdoff_scale_permille,
                )
                .map_err(|error| SpecificationError::wrap("phase modulation", error))?,
            );
        }
        Ok(policy)
    }

    fn build(&self, mode: CompactionMode) -> Result<CompactionParameters, SpecificationError> {
        let mut builder = CompactionParameters::builder(self.theta_r, self.policy()?)
            .mode(mode)
            .audit(self.audit.core())
            .evidence_sample(self.evidence_sample)
            .census(self.census);
        for operator in &self.operators {
            builder = builder.operator(operator.core());
        }
        builder
            .build()
            .map_err(|error| SpecificationError::wrap("compaction parameters", error))
    }

    pub fn parameters(&self) -> Result<CompactionParameters, SpecificationError> {
        let validated = self.build(CompactionMode::Apply)?;
        match self.mode.core() {
            CompactionMode::Apply => Ok(validated),
            mode => self.build(mode),
        }
    }

    pub fn vcp_configuration(&self) -> Result<VcpCompactingConfiguration, SpecificationError> {
        let clock = match self.clock {
            Some(ClockSpecification::PerHead) => CompactionClock::PerHead,
            Some(ClockSpecification::Synchronous) => CompactionClock::Synchronous,
            None => return Err(SpecificationError::new("a VCP compaction needs a clock")),
        };
        let mut configuration = VcpCompactingConfiguration::new(clock, self.parameters()?);
        if let Some(limits) = self.acceptance_limits {
            let valid = [limits.max_mean_acceptance_shift, limits.max_mean_variance_shift]
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0);
            if !valid {
                return Err(SpecificationError::new(
                    "acceptance limits must be finite and non-negative",
                ));
            }
            configuration = configuration.with_acceptance_limits(AcceptanceLimits::new(
                limits.max_mean_acceptance_shift,
                limits.max_mean_variance_shift,
            ));
        }
        if let Some(probe) = self.disagreement_probe {
            if probe.every_episodes == 0 || probe.sample == 0 {
                return Err(SpecificationError::new(
                    "a disagreement probe needs a positive period and sample",
                ));
            }
            configuration = configuration
                .with_disagreement_probe(DisagreementProbe::new(probe.every_episodes, probe.sample));
        }
        Ok(configuration)
    }

    pub fn validate_for(&self, family: Family) -> Result<(), SpecificationError> {
        let parameters = self.parameters()?;
        match family {
            Family::Vcp => {
                self.vcp_configuration()?;
            }
            Family::Acs2 | Family::Acs2Er | Family::Episodic => {
                if self.clock == Some(ClockSpecification::Synchronous) {
                    return Err(SpecificationError::new(
                        "a single-head agent has one clock; use per_head or leave clock out",
                    ));
                }
                let rejected = [
                    (self.acceptance_limits.is_some(), "acceptance_limits"),
                    (self.disagreement_probe.is_some(), "disagreement_probe"),
                ]
                .into_iter()
                .find_map(|(present, field)| present.then_some(field));
                if let Some(field) = rejected {
                    return Err(SpecificationError::new(format!(
                        "{field} applies to ACS2VCP only"
                    )));
                }
                if family == Family::Acs2 {
                    acs2_agents::acs2::compacting::CompactingConfiguration::new(parameters)
                        .map_err(|error| SpecificationError::wrap("ACS2 compaction", error))?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowEntrySpecification {
    pub label: String,
    pub operators: Vec<OperatorSpecification>,
    pub strength: StrengthSpecification,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowSpecification {
    pub theta_r: f64,
    pub evidence_sample: usize,
    pub audit: Option<ThresholdsSpecification>,
    pub census: bool,
    pub entries: Vec<ShadowEntrySpecification>,
}

pub struct ShadowEntry {
    pub label: String,
    pub parameters: CompactionParameters,
    pub strength: CompactionStrength,
}

impl ShadowSpecification {
    pub fn from_json(text: &str) -> Result<Self, SpecificationError> {
        let specification: Self = serde_json::from_str(text)
            .map_err(|error| SpecificationError::wrap("compaction shadow specification", error))?;
        specification.entries()?;
        Ok(specification)
    }

    pub fn from_path(path: &Path) -> Result<Self, SpecificationError> {
        Self::from_json(&read(path)?)
            .map_err(|error| SpecificationError::new(format!("{}: {error}", path.display())))
    }

    pub fn leak(self) -> &'static Self {
        Box::leak(Box::new(self))
    }

    pub fn echo(&self) -> Value {
        serde_json::to_value(self).expect("a specification serializes")
    }

    pub fn entries(&self) -> Result<Vec<ShadowEntry>, SpecificationError> {
        if self.entries.is_empty() {
            return Err(SpecificationError::new("a shadow specification needs an entry"));
        }
        let audit = self
            .audit
            .map_or(AuditSpecification::Skip, AuditSpecification::Observe);
        let mut result: Vec<ShadowEntry> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            if entry.label.is_empty() || result.iter().any(|seen| seen.label == entry.label) {
                return Err(SpecificationError::new(format!(
                    "shadow entry labels must be non-empty and unique, got {:?}",
                    entry.label
                )));
            }
            let strength = entry.strength.core();
            let policy = fixed_strength_policy(Never, strength)
                .map_err(|error| SpecificationError::wrap("shadow strength", error))?;
            let mut builder = CompactionParameters::builder(self.theta_r, policy)
                .mode(CompactionMode::Shadow)
                .audit(audit.core())
                .evidence_sample(self.evidence_sample)
                .census(self.census);
            for operator in &entry.operators {
                builder = builder.operator(operator.core());
            }
            let parameters = builder.build().map_err(|error| {
                SpecificationError::wrap(&format!("shadow entry {}", entry.label), error)
            })?;
            result.push(ShadowEntry {
                label: entry.label.clone(),
                parameters,
                strength,
            });
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn full() -> Value {
        json!({
            "mode": "apply",
            "clock": "per_head",
            "theta_r": 0.9,
            "operators": [
                {"kind": "mature_subsumption", "guarded": true},
                {"kind": "idle_pruning", "min_experience": 5}
            ],
            "trigger": {"kind": "cost_benefit", "prior_yield_permille": 200,
                "yield_floor_permille": 20, "margin_permille": 1000},
            "strength": {"value_tolerance": 0.0, "maturity_experience": 20, "idle_audits": 2},
            "strength_bounds": {
                "gentlest": {"value_tolerance": 0.0, "maturity_experience": 40, "idle_audits": 4},
                "boldest": {"value_tolerance": 0.05, "maturity_experience": 10, "idle_audits": 1},
                "step": {"value_tolerance": 0.01, "maturity_experience": 5, "idle_audits": 1}
            },
            "feedback": {"low_yield_permille": 20, "high_yield_permille": 200,
                "restored_bound_permille": 100, "minimum_holdoff": 500,
                "initial_holdoff": 1000, "maximum_holdoff": 8000},
            "phase_modulation": {"creation_rate_ceiling_per_thousand": 50,
                "coverage_floor_permille": 900, "strength_steps": 1, "holdoff_scale_permille": 2000},
            "audit": {"policy": "enforce", "max_coverage_loss": 0.0,
                "max_no_change_coverage_loss": 0.0, "max_value_change": 0.0,
                "max_greedy_change_fraction": 0.0},
            "evidence_sample": 500,
            "census": true,
            "acceptance_limits": {"max_mean_acceptance_shift": 0.1, "max_mean_variance_shift": 5.0},
            "disagreement_probe": {"every_episodes": 100, "sample": 200},
            "max_records_per_row": 50
        })
    }

    fn parse(value: &Value) -> Result<CompactionSpecification, SpecificationError> {
        CompactionSpecification::from_json(&value.to_string())
    }

    fn with(mutation: impl FnOnce(&mut Value)) -> Value {
        let mut value = full();
        mutation(&mut value);
        value
    }

    #[test]
    fn a_full_specification_parses_into_the_core_types_and_echoes_every_field() {
        let specification = parse(&full()).unwrap();
        let parameters = specification.parameters().unwrap();
        assert_eq!(parameters.mode(), CompactionMode::Apply);
        assert_eq!(parameters.operators().len(), 2);
        assert_eq!(parameters.evidence_sample(), 500);
        assert!(parameters.audit().is_enforced());
        assert_eq!(parameters.theta_r(), 0.9);
        assert!(parameters.records_census());
        let policy = parameters.policy();
        assert_eq!(policy.holdoff_interactions(), 1000);
        assert!(matches!(policy.trigger(), BaseTrigger::CostBenefit(_)));
        specification.validate_for(Family::Vcp).unwrap();
        assert_eq!(specification.echo(), full());
    }

    #[test]
    fn absent_optional_features_are_echoed_as_null() {
        let value = with(|value| {
            let object = value.as_object_mut().unwrap();
            for key in [
                "clock",
                "strength_bounds",
                "feedback",
                "phase_modulation",
                "acceptance_limits",
                "disagreement_probe",
            ] {
                object.remove(key);
            }
            value["audit"] = json!({"policy": "skip"});
            value["operators"] = json!([{"kind": "cracs_subsumption", "guarded": false}]);
            value["trigger"] = json!({"kind": "periodic", "every_interactions": 1000});
        });
        let specification = parse(&value).unwrap();
        let echo = specification.echo();
        assert_eq!(echo["feedback"], Value::Null);
        assert_eq!(echo["clock"], Value::Null);
        assert_eq!(echo["audit"], json!({"policy": "skip"}));
        specification.validate_for(Family::Episodic).unwrap();
    }

    #[test]
    fn unknown_and_missing_fields_are_rejected() {
        let rejected = [
            with(|value| value["surprise"] = json!(1)),
            with(|value| value["operators"][0]["surprise"] = json!(1)),
            with(|value| value["trigger"]["surprise"] = json!(1)),
            with(|value| value["audit"]["surprise"] = json!(1)),
            with(|value| value["strength"]["surprise"] = json!(1)),
            with(|value| value["strength_bounds"]["step"]["surprise"] = json!(1)),
            with(|value| value["operators"][0]["kind"] = json!("magic")),
            with(|value| value["mode"] = json!("sometimes")),
            with(|value| {
                value.as_object_mut().unwrap().remove("theta_r");
            }),
            with(|value| {
                value["audit"].as_object_mut().unwrap().remove("max_value_change");
            }),
        ];
        for value in rejected {
            assert!(parse(&value).is_err(), "{value}");
        }
    }

    #[test]
    fn invalid_values_surface_the_builder_errors() {
        let cases = [
            (
                with(|value| value["trigger"] = json!({"kind": "periodic", "every_interactions": 0})),
                "period",
            ),
            (
                with(|value| {
                    value["trigger"] = json!({"kind": "growth", "ratio_permille": 1000, "minimum_reference": 10})
                }),
                "growth ratio",
            ),
            (with(|value| value["operators"] = json!([])), "at least one operator"),
            (
                with(|value| {
                    value["operators"] = json!([
                        {"kind": "mature_subsumption", "guarded": true},
                        {"kind": "mature_subsumption", "guarded": false}
                    ])
                }),
                "more than once",
            ),
            (with(|value| value["theta_r"] = json!(1.5)), "theta_r"),
            (with(|value| value["evidence_sample"] = json!(0)), "evidence sample"),
            (
                with(|value| value["audit"]["max_value_change"] = json!(-1.0)),
                "audit thresholds",
            ),
            (
                with(|value| {
                    value["strength_bounds"]["gentlest"]["maturity_experience"] = json!(1)
                }),
                "strength bounds",
            ),
            (
                with(|value| value["feedback"]["low_yield_permille"] = json!(900)),
                "feedback",
            ),
            (
                with(|value| value["phase_modulation"]["holdoff_scale_permille"] = json!(10)),
                "phase",
            ),
            (
                with(|value| {
                    value.as_object_mut().unwrap().remove("strength_bounds");
                }),
                "needs strength_bounds",
            ),
        ];
        for (value, fragment) in cases {
            let error = parse(&value).unwrap_err().to_string();
            assert!(error.contains(fragment), "{fragment} not in {error}");
        }
    }

    #[test]
    fn family_rules_reject_inapplicable_or_missing_features() {
        let specification = parse(&full()).unwrap();
        for family in [Family::Acs2, Family::Acs2Er, Family::Episodic] {
            let error = specification.validate_for(family).unwrap_err().to_string();
            assert!(error.contains("ACS2VCP only"), "{error}");
        }
        let synchronous = parse(&with(|value| {
            value["clock"] = json!("synchronous");
            for key in ["acceptance_limits", "disagreement_probe"] {
                value.as_object_mut().unwrap().remove(key);
            }
        }))
        .unwrap();
        let error = synchronous.validate_for(Family::Episodic).unwrap_err().to_string();
        assert!(error.contains("one clock"), "{error}");
        let per_head = parse(&with(|value| {
            for key in ["acceptance_limits", "disagreement_probe"] {
                value.as_object_mut().unwrap().remove(key);
            }
        }))
        .unwrap();
        per_head.validate_for(Family::Episodic).unwrap();
        let no_clock = with(|value| {
            value.as_object_mut().unwrap().remove("clock");
        });
        let error = parse(&no_clock).unwrap().validate_for(Family::Vcp).unwrap_err();
        assert!(error.to_string().contains("clock"));
        let zero_limit = with(|value| {
            value["acceptance_limits"]["max_mean_variance_shift"] = json!(-1.0)
        });
        assert!(parse(&zero_limit).unwrap().validate_for(Family::Vcp).is_err());
        let zero_probe = with(|value| value["disagreement_probe"]["sample"] = json!(0));
        assert!(parse(&zero_probe).unwrap().validate_for(Family::Vcp).is_err());
    }

    #[test]
    fn plain_acs2_refuses_operators_and_audits_that_need_evidence() {
        let mut value = full();
        let object = value.as_object_mut().unwrap();
        for key in [
            "clock",
            "acceptance_limits",
            "disagreement_probe",
            "feedback",
            "phase_modulation",
        ] {
            object.remove(key);
        }
        object.remove("strength_bounds");
        value["strength"]["idle_audits"] = json!(1);
        let specification = parse(&value).unwrap();
        assert!(specification.validate_for(Family::Acs2).is_err());
        value["operators"] = json!([{"kind": "mature_subsumption", "guarded": true}]);
        let specification = parse(&value).unwrap();
        assert!(specification.validate_for(Family::Acs2).is_err());
        value["audit"] = json!({"policy": "skip"});
        value["evidence_sample"] = json!(0);
        parse(&value).unwrap().validate_for(Family::Acs2).unwrap();
    }

    #[test]
    fn off_and_shadow_modes_keep_their_declared_mode() {
        for (name, mode) in [("off", CompactionMode::Off), ("shadow", CompactionMode::Shadow)] {
            let specification = parse(&with(|value| value["mode"] = json!(name))).unwrap();
            assert_eq!(specification.parameters().unwrap().mode(), mode);
        }
    }

    fn shadow_value() -> Value {
        json!({
            "theta_r": 0.9,
            "evidence_sample": 300,
            "audit": {"max_coverage_loss": 0.0, "max_no_change_coverage_loss": 0.0,
                "max_value_change": 0.0, "max_greedy_change_fraction": 0.0},
            "census": true,
            "entries": [
                {"label": "mature", "operators": [{"kind": "mature_subsumption", "guarded": true}],
                    "strength": {"value_tolerance": 0.0, "maturity_experience": 20, "idle_audits": 1}},
                {"label": "idle", "operators": [{"kind": "idle_pruning", "min_experience": 5}],
                    "strength": {"value_tolerance": 0.0, "maturity_experience": 20, "idle_audits": 2}}
            ]
        })
    }

    #[test]
    fn a_shadow_specification_builds_one_shadow_mode_entry_per_label() {
        let specification = ShadowSpecification::from_json(&shadow_value().to_string()).unwrap();
        let entries = specification.entries().unwrap();
        assert_eq!(entries.len(), 2);
        for entry in &entries {
            assert_eq!(entry.parameters.mode(), CompactionMode::Shadow);
            assert_eq!(entry.parameters.evidence_sample(), 300);
            assert!(entry.parameters.audit().thresholds().is_some());
            assert!(!entry.parameters.audit().is_enforced());
        }
        assert_eq!(entries[1].strength.idle_audits, 2);
        assert_eq!(specification.echo(), shadow_value());
    }

    #[test]
    fn bad_shadow_specifications_are_rejected() {
        let mut duplicate = shadow_value();
        duplicate["entries"][1]["label"] = json!("mature");
        let mut empty = shadow_value();
        empty["entries"] = json!([]);
        let mut unknown = shadow_value();
        unknown["entries"][0]["extra"] = json!(true);
        let mut no_evidence = shadow_value();
        no_evidence["evidence_sample"] = json!(0);
        let mut nothing = shadow_value();
        nothing["entries"][0]["operators"] = json!([]);
        for value in [duplicate, empty, unknown, no_evidence, nothing] {
            assert!(
                ShadowSpecification::from_json(&value.to_string()).is_err(),
                "{value}"
            );
        }
    }
}
