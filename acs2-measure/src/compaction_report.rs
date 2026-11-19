use acs2_agents::acs2vcp::compacting::{AcceptanceAuditRecord, AcceptanceSummary, DisagreementRecord};
use acs2_core::compaction::{
    strided_indices, AuditReport, AuditVerdict, Census, Coverage, CoverageCount, MatchingCost,
    CompactionMode, CompactionPolicy, CompactionStrength, ReductionRecord, TriggerObservation,
};
use serde_json::{json, Value};

pub struct HeadCompaction<'a> {
    pub records: &'a [ReductionRecord],
    pub cost: MatchingCost,
    pub policy: Option<&'a CompactionPolicy>,
    pub skips: Option<u64>,
}

pub struct VcpCompaction<'a> {
    pub clock: &'static str,
    pub acceptance: &'a [AcceptanceAuditRecord],
    pub probes: &'a [DisagreementRecord],
    pub acceptance_work: u64,
    pub probe_work: u64,
}

pub struct CompactionView<'a> {
    pub heads: Vec<HeadCompaction<'a>>,
    pub synchronous_policy: Option<&'a CompactionPolicy>,
    pub vcp: Option<VcpCompaction<'a>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PopulationArea {
    pub interactions: u64,
    pub area: u64,
}

impl PopulationArea {
    pub fn observe_episode(&mut self, steps_before: u64, steps_after: u64, population_size: usize) {
        let elapsed = steps_after.saturating_sub(steps_before);
        self.interactions += elapsed;
        self.area += elapsed * population_size as u64;
    }

    pub fn json(&self) -> Value {
        json!({"interactions": self.interactions, "area": self.area,
            "mean_size": (self.interactions > 0).then(|| self.area as f64 / self.interactions as f64),
            "definition": "sum_over_training_episodes_of_end_of_episode_population_classifiers_times_episode_steps"})
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompactionReporter {
    max_records_per_row: usize,
    reported_records: Vec<usize>,
    reported_acceptance: usize,
    reported_probes: usize,
}

fn mode_name(mode: CompactionMode) -> &'static str {
    match mode {
        CompactionMode::Off => "off",
        CompactionMode::Shadow => "shadow",
        CompactionMode::Apply => "apply",
    }
}

fn strength_json(strength: &CompactionStrength) -> Value {
    json!({"value_tolerance": strength.value_tolerance,
        "maturity_experience": strength.maturity_experience, "idle_audits": strength.idle_audits})
}

fn observation_json(observation: &TriggerObservation) -> Value {
    json!({"interactions": observation.interactions, "population_size": observation.population_size,
        "match_formations_since_reduction": observation.match_formations_since_reduction,
        "classifier_tests_since_reduction": observation.classifier_tests_since_reduction,
        "rules_created_since_reduction": observation.rules_created_since_reduction,
        "coverage_permille": observation.coverage_permille})
}

fn census_json(census: &Census) -> Value {
    json!({"rules": census.rules, "numerosity": census.numerosity, "reliable": census.reliable,
        "marked": census.marked, "change_anticipating": census.change_anticipating,
        "goal_specific": census.goal_specific, "value_carriers": census.value_carriers,
        "experience": census.experience.map(|experience| json!({"minimum": experience.minimum,
            "lower_quartile": experience.lower_quartile, "median": experience.median,
            "upper_quartile": experience.upper_quartile, "maximum": experience.maximum,
            "mean": experience.mean})),
        "mean_specificity": census.mean_specificity,
        "reliable_mean_specificity": census.reliable_mean_specificity, "work": census.work})
}

fn count_json(count: &CoverageCount) -> Value {
    json!({"transitions": count.transitions, "covered": count.covered})
}

fn coverage_json(coverage: &Coverage) -> Value {
    json!({"all": count_json(&coverage.all), "state_changing": count_json(&coverage.state_changing),
        "no_change": count_json(&coverage.no_change)})
}

pub fn audit_json(report: &AuditReport, verdict: Option<&AuditVerdict>) -> Value {
    json!({"coverage_before": coverage_json(&report.coverage_before),
        "coverage_after": coverage_json(&report.coverage_after),
        "coverage_loss": report.coverage_loss(),
        "state_changing_coverage_loss": report.state_changing_coverage_loss(),
        "no_change_coverage_loss": report.no_change_coverage_loss(),
        "states": report.states, "value_pairs": report.value_pairs,
        "max_value_change": report.max_value_change, "mean_value_change": report.mean_value_change,
        "greedy_changes": report.greedy_changes,
        "greedy_change_fraction": report.greedy_change_fraction(), "work": report.work,
        "verdict": verdict.map(|verdict| json!({"accepted": verdict.accepted(),
            "violations": verdict.violations.iter().map(|violation| format!("{violation:?}")).collect::<Vec<_>>()}))})
}

pub fn record_json(head: usize, record: &ReductionRecord) -> Value {
    json!({"head": head, "sequence": record.sequence, "mode": mode_name(record.mode),
        "interaction": record.interaction, "observation": observation_json(&record.observation),
        "strength": strength_json(&record.strength), "evidence_size": record.evidence_size,
        "size_before": record.size_before, "size_candidate": record.size_candidate,
        "size_after": record.size_after,
        "removals": {"subsumed": record.removals.subsumed, "idle": record.removals.idle,
            "low_quality": record.removals.low_quality, "evicted": record.removals.evicted,
            "total": record.removals.total()},
        "value_vetoes": record.value_vetoes,
        "stages": record.stages.iter().map(|stage| json!({"operator": format!("{:?}", stage.operator),
            "removed": stage.removed, "work": stage.work, "value_vetoes": stage.value_vetoes})).collect::<Vec<_>>(),
        "work": {"operator": record.operator_work, "audit": record.audit_work,
            "check": record.check_work, "census": record.census_work, "charged": record.charged_work},
        "audit": record.audit.as_ref().map(|report| audit_json(report, record.verdict.as_ref())),
        "check_vetoed": record.check_vetoed, "accepted": record.accepted, "applied": record.applied,
        "restoration": record.restoration.map(|restoration| json!({"removed": restoration.removed,
            "restored": restoration.restored, "fraction": restoration.fraction()})),
        "census_before": record.census_before.as_ref().map(census_json),
        "census_candidate": record.census_candidate.as_ref().map(census_json),
        "feedback": {"population_before": record.feedback.population_before,
            "population_after": record.feedback.population_after,
            "work_classifier_tests": record.feedback.work_classifier_tests,
            "rejected_by_guard": record.feedback.rejected_by_guard,
            "restored_permille": record.feedback.restored_permille}})
}

fn policy_json(policy: &CompactionPolicy) -> Value {
    json!({"fires": policy.fires(), "reductions_reported": policy.reductions_reported(),
        "strength": strength_json(&policy.strength()),
        "holdoff_interactions": policy.holdoff_interactions(),
        "last_yield_permille": policy.last_yield_permille(),
        "last_fire_interaction": policy.last_fire_interaction(),
        "learning_active": policy.learning_active()})
}

fn summary_json(summary: &AcceptanceSummary) -> Value {
    json!({"mean_variance": summary.mean_variance, "mean_acceptance": summary.mean_acceptance})
}

fn cost_json(cost: &MatchingCost) -> Value {
    json!({"match_formations": cost.match_formations, "classifier_tests": cost.classifier_tests})
}

impl CompactionReporter {
    pub fn new(max_records_per_row: usize) -> Self {
        Self {
            max_records_per_row,
            ..Self::default()
        }
    }

    pub fn report(&mut self, view: &CompactionView<'_>, area: &PopulationArea) -> Value {
        self.reported_records.resize(view.heads.len(), 0);
        let mut totals = Totals::default();
        let mut fresh: Vec<(u64, usize, &ReductionRecord)> = Vec::new();
        let mut heads = Vec::with_capacity(view.heads.len());
        for (index, head) in view.heads.iter().enumerate() {
            totals.cost.match_formations += head.cost.match_formations;
            totals.cost.classifier_tests += head.cost.classifier_tests;
            let mut applied = 0usize;
            for record in head.records {
                totals.add(record);
                applied += usize::from(record.applied);
            }
            let from = self.reported_records[index].min(head.records.len());
            fresh.extend(
                head.records[from..]
                    .iter()
                    .map(|record| (record.interaction, index, record)),
            );
            self.reported_records[index] = head.records.len();
            heads.push(json!({"head": index, "reductions": head.records.len(), "applied": applied,
                "skips": head.skips, "training_cost": cost_json(&head.cost),
                "policy": head.policy.map(policy_json)}));
        }
        fresh.sort_by_key(|(interaction, head, record)| (*interaction, *head, record.sequence));
        let new_records = fresh.len();
        let kept: Vec<Value> = if new_records <= self.max_records_per_row {
            fresh
                .iter()
                .map(|(_, head, record)| record_json(*head, record))
                .collect()
        } else {
            strided_indices(new_records, self.max_records_per_row)
                .into_iter()
                .map(|position| record_json(fresh[position].1, fresh[position].2))
                .collect()
        };
        let mut report = json!({
            "reductions": totals.reductions, "accepted": totals.accepted,
            "applied": totals.applied, "rejected": totals.reductions - totals.accepted,
            "removed_applied": totals.removed_applied, "removed_candidates": totals.removed_candidates,
            "work": {"operator": totals.operator, "audit": totals.audit, "check": totals.check,
                "census": totals.census, "charged": totals.charged,
                "uncharged": totals.operator + totals.audit + totals.check + totals.census - totals.charged,
                "unit": "classifier_perception_tests_not_in_match_counters"},
            "training_cost": cost_json(&totals.cost),
            "training_cost_definition": "derived_from_update_counts_and_population_sizes_not_measured",
            "population_size_area": area.json(),
            "heads": heads,
            "synchronous_policy": view.synchronous_policy.map(policy_json),
            "records_scope": "reductions_since_previous_row",
            "records_new": new_records, "records_omitted": new_records - kept.len(),
            "records": kept});
        if let Some(vcp) = &view.vcp {
            let acceptance = &vcp.acceptance[self.reported_acceptance.min(vcp.acceptance.len())..];
            let probes = &vcp.probes[self.reported_probes.min(vcp.probes.len())..];
            self.reported_acceptance = vcp.acceptance.len();
            self.reported_probes = vcp.probes.len();
            report["vcp"] = json!({"clock": vcp.clock,
                "acceptance_audits_total": vcp.acceptance.len(),
                "acceptance_vetoes_total": vcp.acceptance.iter().filter(|audit| audit.vetoed).count(),
                "acceptance_work": vcp.acceptance_work, "probe_work": vcp.probe_work,
                "acceptance_audits": acceptance.iter().map(|audit| json!({
                    "head": audit.head, "sequence": audit.sequence, "interaction": audit.interaction,
                    "samples": audit.samples, "before": summary_json(&audit.before),
                    "after": summary_json(&audit.after), "vetoed": audit.vetoed, "work": audit.work})).collect::<Vec<_>>(),
                "disagreement_probes": probes.iter().map(|probe| json!({
                    "episodes": probe.episodes, "interactions": probe.interactions,
                    "samples": probe.samples, "summary": summary_json(&probe.summary),
                    "head_sizes": probe.head_sizes, "work": probe.work})).collect::<Vec<_>>()});
        }
        report
    }
}

#[derive(Default)]
struct Totals {
    reductions: usize,
    accepted: usize,
    applied: usize,
    removed_applied: usize,
    removed_candidates: usize,
    operator: u64,
    audit: u64,
    check: u64,
    census: u64,
    charged: u64,
    cost: MatchingCost,
}

impl Totals {
    fn add(&mut self, record: &ReductionRecord) {
        self.reductions += 1;
        self.accepted += usize::from(record.accepted);
        self.applied += usize::from(record.applied);
        self.removed_applied += record.size_before - record.size_after;
        self.removed_candidates += record.size_before - record.size_candidate;
        self.operator += record.operator_work;
        self.audit += record.audit_work;
        self.check += record.check_work;
        self.census += record.census_work;
        self.charged += record.charged_work;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_area_is_the_end_of_episode_size_weighted_by_the_episode_steps() {
        let mut area = PopulationArea::default();
        area.observe_episode(0, 10, 4);
        area.observe_episode(10, 25, 2);
        assert_eq!(area.interactions, 25);
        assert_eq!(area.area, 40 + 30);
        assert_eq!(area.json()["mean_size"], json!(70.0 / 25.0));
        assert_eq!(area.json()["area"], json!(70));
    }

    #[test]
    fn an_empty_view_reports_zero_reductions_and_no_records() {
        let mut reporter = CompactionReporter::new(5);
        let view = CompactionView {
            heads: Vec::new(),
            synchronous_policy: None,
            vcp: None,
        };
        let report = reporter.report(&view, &PopulationArea::default());
        assert_eq!(report["reductions"], 0);
        assert_eq!(report["records"], json!([]));
        assert_eq!(report["records_scope"], "reductions_since_previous_row");
    }
}
