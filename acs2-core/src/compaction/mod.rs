pub mod audit;
pub mod capacity;
pub mod census;
pub mod controller;
pub mod cost;
pub mod cracs;
pub mod dynamic_thresholds;
pub mod evidence;
pub mod evidence_source;
pub mod idle_pruning;
pub mod ledger;
pub mod mature_subsumption;
pub mod outcome;
pub mod rule;
pub mod rule_key;
pub mod single_head;
pub mod trigger;
pub mod value_guard;

#[cfg(test)]
mod test_support;

pub use audit::{
    audit_reduction, AuditReport, AuditThresholds, AuditVerdict, Coverage, CoverageCount,
    Violation,
};
pub use capacity::{evict_to_capacity, evict_to_capacity_population, CapacityConfiguration};
pub use census::{population_census, Census, ExperienceSummary};
pub use controller::{
    fixed_strength_policy, no_candidate_check, AuditPolicy, CandidateCheck, CheckResult,
    CompactionController, CompactionMode, CompactionParameterError, CompactionParameters,
    CompactionParametersBuilder, CompactionSignals, Operator, OperatorKind, ReductionEngine,
    ReductionRecord, ReductionReport, ReductionRequest, RemovalCounts, SignalWindow, StageRecord,
    TaskGeometry, COMPACTION_STREAM,
};
pub use cost::{EpisodeFormations, MatchingCost};
pub use cracs::{
    compact, compact_population, is_subsumed, reliable_mean_specificity, CompactionConfiguration,
    CompactionOutcome, Decision, FilterReason,
};
pub use dynamic_thresholds::{
    prune_by_thresholds, prune_by_thresholds_population, DynamicThresholds,
};
pub use evidence_source::{strided_indices, subsample_indices, EvidenceSource, NoEvidence};
pub use idle_pruning::{idle_flags, IdlePruningConfiguration, IdleTracker};
pub use ledger::{RemovalLedger, Restoration};
pub use mature_subsumption::{
    is_mature_subsumed, mature_subsumption, mature_subsumption_population,
    MatureSubsumptionConfiguration,
};
pub use outcome::{ReductionOutcome, RemovalReason, Verdict};
pub use rule::{CompactableRule, ValuedRule};
pub use rule_key::RuleKey;
pub use single_head::SingleHeadCompaction;
pub use trigger::{
    BaseTrigger, CompactionPolicy, CompactionStrength, ReductionFeedback, StrengthBounds,
    TriggerCondition, TriggerDecision, TriggerObservation,
};
pub use value_guard::{
    absorb, compact_guarded, compact_guarded_population, ValueGuard,
};
