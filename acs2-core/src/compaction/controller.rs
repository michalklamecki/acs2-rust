use core::fmt;
use core::ops::Range;
use std::error::Error;

use crate::compaction::audit::{audit_reduction, AuditReport, AuditThresholds, AuditVerdict};
use crate::compaction::capacity::{evict_to_capacity, CapacityConfiguration};
use crate::compaction::census::{population_census, Census};
use crate::compaction::cracs::CompactionConfiguration;
use crate::compaction::dynamic_thresholds::{prune_by_thresholds, DynamicThresholds};
use crate::compaction::evidence_source::EvidenceSource;
use crate::compaction::idle_pruning::{IdlePruningConfiguration, IdleTracker};
use crate::compaction::ledger::{RemovalLedger, Restoration};
use crate::compaction::mature_subsumption::{mature_subsumption, MatureSubsumptionConfiguration};
use crate::compaction::outcome::{ReductionOutcome, RemovalReason, Verdict};
use crate::compaction::trigger::{
    BaseTrigger, CompactionPolicy, CompactionStrength, Never, ReductionFeedback, StrengthBounds,
    StrengthStep, TriggerDecision, TriggerObservation, TriggerParameterError,
};
use crate::compaction::value_guard::{compact_guarded, ValueGuard};
use crate::representation::classifier::Classifier;
use crate::representation::population::Population;
use crate::support::knowledge::Transition;
use crate::support::rng::RandomSource;

pub const COMPACTION_STREAM: u64 = 11;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompactionMode {
    Off,
    Shadow,
    Apply,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OperatorKind {
    MatureSubsumption,
    CracsSubsumption,
    IdlePruning,
    DynamicThresholds,
    Capacity,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Operator {
    MatureSubsumption { guarded: bool },
    CracsSubsumption { guarded: bool },
    IdlePruning { min_experience: u32 },
    DynamicThresholds(DynamicThresholds),
    Capacity(CapacityConfiguration),
}

impl Operator {
    pub fn kind(&self) -> OperatorKind {
        match self {
            Self::MatureSubsumption { .. } => OperatorKind::MatureSubsumption,
            Self::CracsSubsumption { .. } => OperatorKind::CracsSubsumption,
            Self::IdlePruning { .. } => OperatorKind::IdlePruning,
            Self::DynamicThresholds(_) => OperatorKind::DynamicThresholds,
            Self::Capacity(_) => OperatorKind::Capacity,
        }
    }

    pub fn needs_evidence(&self) -> bool {
        matches!(self, Self::IdlePruning { .. } | Self::Capacity(_))
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AuditPolicy {
    Skip,
    Observe(AuditThresholds),
    Enforce(AuditThresholds),
}

impl AuditPolicy {
    pub fn thresholds(&self) -> Option<&AuditThresholds> {
        match self {
            Self::Skip => None,
            Self::Observe(thresholds) | Self::Enforce(thresholds) => Some(thresholds),
        }
    }

    pub fn is_enforced(&self) -> bool {
        matches!(self, Self::Enforce(_))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompactionParameterError {
    NoOperators,
    DuplicateOperator(OperatorKind),
    ReliabilityOutOfRange,
    InvalidAuditThresholds,
    EvidenceRequired,
}

impl fmt::Display for CompactionParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoOperators => write!(formatter, "a compacting mode needs at least one operator"),
            Self::DuplicateOperator(kind) => {
                write!(formatter, "operator {kind:?} appears more than once")
            }
            Self::ReliabilityOutOfRange => write!(formatter, "theta_r must lie in (0, 1]"),
            Self::InvalidAuditThresholds => {
                write!(formatter, "audit thresholds must be finite and non-negative")
            }
            Self::EvidenceRequired => write!(
                formatter,
                "an evidence operator or an audit needs a positive evidence sample"
            ),
        }
    }
}

impl Error for CompactionParameterError {}

pub fn fixed_strength_policy(
    trigger: impl Into<BaseTrigger>,
    strength: CompactionStrength,
) -> Result<CompactionPolicy, TriggerParameterError> {
    let step = StrengthStep::new(0.0, 0, 0)?;
    let bounds = StrengthBounds::new(strength, strength, step)?;
    Ok(CompactionPolicy::new(trigger, bounds, strength))
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactionParameters {
    mode: CompactionMode,
    operators: Vec<Operator>,
    policy: CompactionPolicy,
    audit: AuditPolicy,
    evidence_sample: usize,
    theta_r: f64,
    record_census: bool,
}

impl CompactionParameters {
    pub fn off() -> Self {
        let inert = CompactionStrength {
            value_tolerance: 0.0,
            maturity_experience: 0,
            idle_audits: 0,
        };
        Self {
            mode: CompactionMode::Off,
            operators: Vec::new(),
            policy: fixed_strength_policy(Never, inert)
                .expect("an inert fixed strength is always valid"),
            audit: AuditPolicy::Skip,
            evidence_sample: 0,
            theta_r: 1.0,
            record_census: false,
        }
    }

    pub fn builder(theta_r: f64, policy: CompactionPolicy) -> CompactionParametersBuilder {
        CompactionParametersBuilder {
            parameters: Self {
                mode: CompactionMode::Apply,
                operators: Vec::new(),
                policy,
                audit: AuditPolicy::Skip,
                evidence_sample: 0,
                theta_r,
                record_census: true,
            },
        }
    }

    pub fn mode(&self) -> CompactionMode {
        self.mode
    }

    pub fn operators(&self) -> &[Operator] {
        &self.operators
    }

    pub fn policy(&self) -> &CompactionPolicy {
        &self.policy
    }

    pub fn audit(&self) -> &AuditPolicy {
        &self.audit
    }

    pub fn evidence_sample(&self) -> usize {
        self.evidence_sample
    }

    pub fn theta_r(&self) -> f64 {
        self.theta_r
    }

    pub fn records_census(&self) -> bool {
        self.record_census
    }

    pub fn evidence_operator(&self) -> Option<OperatorKind> {
        self.operators
            .iter()
            .find(|operator| operator.needs_evidence())
            .map(Operator::kind)
    }

    fn validated(self) -> Result<Self, CompactionParameterError> {
        if self.mode == CompactionMode::Off {
            return Ok(self);
        }
        if self.operators.is_empty() {
            return Err(CompactionParameterError::NoOperators);
        }
        for (position, operator) in self.operators.iter().enumerate() {
            let repeated = self.operators[..position]
                .iter()
                .any(|earlier| earlier.kind() == operator.kind());
            if repeated {
                return Err(CompactionParameterError::DuplicateOperator(operator.kind()));
            }
        }
        if !(self.theta_r > 0.0 && self.theta_r <= 1.0) {
            return Err(CompactionParameterError::ReliabilityOutOfRange);
        }
        if let Some(thresholds) = self.audit.thresholds() {
            let valid = [
                thresholds.max_coverage_loss,
                thresholds.max_no_change_coverage_loss,
                thresholds.max_value_change,
                thresholds.max_greedy_change_fraction,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0);
            if !valid {
                return Err(CompactionParameterError::InvalidAuditThresholds);
            }
        }
        let evidence_wanted = self.operators.iter().any(Operator::needs_evidence)
            || self.audit.thresholds().is_some();
        if evidence_wanted && self.evidence_sample == 0 {
            return Err(CompactionParameterError::EvidenceRequired);
        }
        Ok(self)
    }
}

#[derive(Clone, Debug)]
pub struct CompactionParametersBuilder {
    parameters: CompactionParameters,
}

impl CompactionParametersBuilder {
    pub fn mode(mut self, mode: CompactionMode) -> Self {
        self.parameters.mode = mode;
        self
    }

    pub fn operator(mut self, operator: Operator) -> Self {
        self.parameters.operators.push(operator);
        self
    }

    pub fn audit(mut self, audit: AuditPolicy) -> Self {
        self.parameters.audit = audit;
        self
    }

    pub fn evidence_sample(mut self, size: usize) -> Self {
        self.parameters.evidence_sample = size;
        self
    }

    pub fn census(mut self, record: bool) -> Self {
        self.parameters.record_census = record;
        self
    }

    pub fn build(self) -> Result<CompactionParameters, CompactionParameterError> {
        self.parameters.validated()
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct TaskGeometry {
    pub actions: usize,
    pub goal_attributes: Range<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CompactionSignals {
    pub interactions: u64,
    pub population_size: usize,
    pub match_formations: u64,
    pub classifier_tests: u64,
    pub rules_created: Option<u64>,
    pub coverage_permille: Option<u32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SignalWindow {
    formations: u64,
    tests: u64,
    created: u64,
}

impl SignalWindow {
    pub fn observation(&self, signals: &CompactionSignals) -> TriggerObservation {
        TriggerObservation {
            interactions: signals.interactions,
            population_size: signals.population_size,
            match_formations_since_reduction: signals
                .match_formations
                .saturating_sub(self.formations),
            classifier_tests_since_reduction: signals.classifier_tests.saturating_sub(self.tests),
            rules_created_since_reduction: signals
                .rules_created
                .map(|total| total.saturating_sub(self.created)),
            coverage_permille: signals.coverage_permille,
        }
    }

    pub fn reset(&mut self, signals: &CompactionSignals) {
        self.formations = signals.match_formations;
        self.tests = signals.classifier_tests;
        if let Some(created) = signals.rules_created {
            self.created = created;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct RemovalCounts {
    pub subsumed: usize,
    pub idle: usize,
    pub low_quality: usize,
    pub evicted: usize,
}

impl RemovalCounts {
    pub fn total(&self) -> usize {
        self.subsumed + self.idle + self.low_quality + self.evicted
    }

    fn add(&mut self, outcome: &ReductionOutcome) {
        for verdict in &outcome.decisions {
            if let Verdict::Removed(reason) = verdict {
                match reason {
                    RemovalReason::Subsumed { .. } => self.subsumed += 1,
                    RemovalReason::Idle { .. } => self.idle += 1,
                    RemovalReason::LowQuality => self.low_quality += 1,
                    RemovalReason::Evicted => self.evicted += 1,
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StageRecord {
    pub operator: OperatorKind,
    pub removed: usize,
    pub work: u64,
    pub value_vetoes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CheckResult {
    pub vetoes: bool,
    pub work: u64,
}

impl CheckResult {
    pub fn passed() -> Self {
        Self {
            vetoes: false,
            work: 0,
        }
    }
}

pub type CandidateCheck<'a, const N: usize> =
    dyn FnMut(&Population<N>, &Population<N>, &[Transition<N>]) -> CheckResult + 'a;

pub fn no_candidate_check<const N: usize>(
    _before: &Population<N>,
    _candidate: &Population<N>,
    _evidence: &[Transition<N>],
) -> CheckResult {
    CheckResult::passed()
}

#[derive(Clone, PartialEq, Debug)]
pub struct ReductionRecord {
    pub sequence: u64,
    pub mode: CompactionMode,
    pub interaction: u64,
    pub observation: TriggerObservation,
    pub strength: CompactionStrength,
    pub evidence_size: usize,
    pub size_before: usize,
    pub size_candidate: usize,
    pub size_after: usize,
    pub removals: RemovalCounts,
    pub value_vetoes: usize,
    pub stages: Vec<StageRecord>,
    pub operator_work: u64,
    pub audit_work: u64,
    pub check_work: u64,
    pub census_work: u64,
    pub charged_work: u64,
    pub audit: Option<AuditReport>,
    pub verdict: Option<AuditVerdict>,
    pub check_vetoed: bool,
    pub accepted: bool,
    pub applied: bool,
    pub restoration: Option<Restoration>,
    pub census_before: Option<Census>,
    pub census_candidate: Option<Census>,
    pub feedback: ReductionFeedback,
}

pub struct ReductionReport<const N: usize> {
    pub feedback: ReductionFeedback,
    pub candidate: Option<Population<N>>,
    pub applied: bool,
}

impl<const N: usize> ReductionReport<N> {
    pub fn replacement(self) -> Option<Population<N>> {
        if self.applied {
            self.candidate
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ReductionRequest {
    pub interaction: u64,
    pub observation: TriggerObservation,
    pub strength: CompactionStrength,
}

#[derive(Clone, PartialEq, Debug)]
pub struct ReductionEngine<const N: usize> {
    parameters: CompactionParameters,
    geometry: TaskGeometry,
    tracker: IdleTracker<N>,
    ledger: RemovalLedger<N>,
    records: Vec<ReductionRecord>,
    last_applied: Option<u64>,
}

impl<const N: usize> ReductionEngine<N> {
    pub fn new(parameters: CompactionParameters, geometry: TaskGeometry) -> Self {
        Self {
            parameters,
            geometry,
            tracker: IdleTracker::new(),
            ledger: RemovalLedger::new(),
            records: Vec::new(),
            last_applied: None,
        }
    }

    pub fn parameters(&self) -> &CompactionParameters {
        &self.parameters
    }

    pub fn tracker(&self) -> &IdleTracker<N> {
        &self.tracker
    }

    pub fn ledger(&self) -> &RemovalLedger<N> {
        &self.ledger
    }

    pub fn records(&self) -> &[ReductionRecord] {
        &self.records
    }

    pub fn reduce<E>(
        &mut self,
        request: &ReductionRequest,
        population: &Population<N>,
        evidence: &E,
        rng: &mut dyn RandomSource,
        check: &mut CandidateCheck<'_, N>,
    ) -> ReductionReport<N>
    where
        E: EvidenceSource<N> + ?Sized,
    {
        let mode = self.parameters.mode;
        assert!(
            mode != CompactionMode::Off,
            "an engine in Off mode must not be asked to reduce"
        );
        let sample = if self.parameters.evidence_sample > 0 {
            evidence.sample(self.parameters.evidence_sample, rng)
        } else {
            Vec::new()
        };
        let restoration = (mode == CompactionMode::Apply)
            .then(|| {
                self.ledger
                    .restoration_since(population, self.last_applied.unwrap_or(0))
            })
            .filter(|restoration| restoration.removed > 0);
        let census_before = self.census(population, &sample);

        let mut staged: Option<Population<N>> = None;
        let mut removed_rules: Vec<Classifier<N>> = Vec::new();
        let mut removals = RemovalCounts::default();
        let mut stages = Vec::new();
        let mut value_vetoes = 0usize;
        let mut operator_work = 0u64;
        let operators = self.parameters.operators.clone();
        for operator in &operators {
            let source = staged.as_ref().unwrap_or(population);
            let outcome = self.run_stage(operator, source, &sample, &request.strength);
            removals.add(&outcome);
            value_vetoes += outcome.value_vetoes;
            operator_work += outcome.work;
            stages.push(StageRecord {
                operator: operator.kind(),
                removed: outcome.removed_count(),
                work: outcome.work,
                value_vetoes: outcome.value_vetoes,
            });
            if outcome.removed_count() > 0 {
                removed_rules.extend(
                    outcome
                        .removed_indices()
                        .into_iter()
                        .map(|index| source.get(index).clone()),
                );
                staged = Some(outcome.apply(source));
            }
        }

        let size_before = population.len();
        let size_candidate = staged.as_ref().map_or(size_before, Population::len);
        let mut audit_report = None;
        let mut verdict = None;
        let mut check_result = CheckResult::passed();
        if let Some(candidate) = staged.as_ref() {
            if let Some(thresholds) = self.parameters.audit.thresholds() {
                let report = audit_reduction(
                    population,
                    candidate,
                    &sample,
                    self.geometry.actions,
                    self.parameters.theta_r,
                );
                verdict = Some(report.verdict(thresholds));
                audit_report = Some(report);
            }
            check_result = check(population, candidate, &sample);
        }
        let audit_work = audit_report.as_ref().map_or(0, |report| report.work);
        let audit_enforced = self.parameters.audit.is_enforced();
        let audit_accepts = !audit_enforced || verdict.as_ref().is_none_or(AuditVerdict::accepted);
        let accepted = audit_accepts && !check_result.vetoes;
        let applied = mode == CompactionMode::Apply && accepted && staged.is_some();

        let census_candidate = match staged.as_ref() {
            Some(candidate) => self.census(candidate, &sample),
            None => census_before.clone(),
        };
        let census_work = census_before.as_ref().map_or(0, |census| census.work)
            + match (staged.as_ref(), census_candidate.as_ref()) {
                (Some(_), Some(census)) => census.work,
                _ => 0,
            };
        let charged_work = operator_work
            + check_result.work
            + if audit_enforced { audit_work } else { 0 };

        if applied {
            self.ledger
                .record(request.interaction, removed_rules.iter());
            self.last_applied = Some(request.interaction);
        }
        let size_after = if applied { size_candidate } else { size_before };
        let feedback = ReductionFeedback {
            population_before: size_before,
            population_after: if accepted { size_candidate } else { size_before },
            work_classifier_tests: charged_work,
            rejected_by_guard: !accepted,
            restored_permille: restoration.map(|restoration| permille(&restoration)),
        };
        let sequence = self.records.len() as u64;
        self.records.push(ReductionRecord {
            sequence,
            mode,
            interaction: request.interaction,
            observation: request.observation,
            strength: request.strength,
            evidence_size: sample.len(),
            size_before,
            size_candidate,
            size_after,
            removals,
            value_vetoes,
            stages,
            operator_work,
            audit_work,
            check_work: check_result.work,
            census_work,
            charged_work,
            audit: audit_report,
            verdict,
            check_vetoed: check_result.vetoes,
            accepted,
            applied,
            restoration,
            census_before,
            census_candidate,
            feedback,
        });
        ReductionReport {
            feedback,
            candidate: staged,
            applied,
        }
    }

    fn census(&self, population: &Population<N>, sample: &[Transition<N>]) -> Option<Census> {
        self.parameters.record_census.then(|| {
            population_census(
                population,
                self.parameters.theta_r,
                self.geometry.goal_attributes.clone(),
                sample,
            )
        })
    }

    fn run_stage(
        &mut self,
        operator: &Operator,
        population: &Population<N>,
        sample: &[Transition<N>],
        strength: &CompactionStrength,
    ) -> ReductionOutcome {
        let theta_r = self.parameters.theta_r;
        let guard = |guarded: bool| guarded.then(|| ValueGuard::new(strength.value_tolerance));
        match operator {
            Operator::MatureSubsumption { guarded } => mature_subsumption(
                population.classifiers(),
                &MatureSubsumptionConfiguration::new(strength.maturity_experience, theta_r),
                guard(*guarded),
            ),
            Operator::CracsSubsumption { guarded } => compact_guarded(
                population.classifiers(),
                &CompactionConfiguration::new(strength.maturity_experience, theta_r),
                guard(*guarded),
            ),
            Operator::IdlePruning { min_experience } => {
                let configuration =
                    IdlePruningConfiguration::new(theta_r, strength.idle_audits, *min_experience);
                let (outcome, next) = self.tracker.observe(population, sample, &configuration);
                self.tracker = next;
                outcome
            }
            Operator::DynamicThresholds(thresholds) => {
                prune_by_thresholds(population.classifiers(), thresholds)
            }
            Operator::Capacity(configuration) => {
                evict_to_capacity(population, sample, configuration)
            }
        }
    }
}

fn permille(restoration: &Restoration) -> u32 {
    if restoration.removed == 0 {
        return 0;
    }
    (restoration.restored as u128 * 1000 / restoration.removed as u128) as u32
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactionController<const N: usize> {
    policy: CompactionPolicy,
    window: SignalWindow,
    engine: ReductionEngine<N>,
    skips: u64,
}

impl<const N: usize> CompactionController<N> {
    pub fn new(parameters: CompactionParameters, geometry: TaskGeometry) -> Self {
        Self {
            policy: parameters.policy.clone(),
            window: SignalWindow::default(),
            engine: ReductionEngine::new(parameters, geometry),
            skips: 0,
        }
    }

    pub fn mode(&self) -> CompactionMode {
        self.engine.parameters.mode
    }

    pub fn parameters(&self) -> &CompactionParameters {
        self.engine.parameters()
    }

    pub fn policy(&self) -> &CompactionPolicy {
        &self.policy
    }

    pub fn tracker(&self) -> &IdleTracker<N> {
        self.engine.tracker()
    }

    pub fn ledger(&self) -> &RemovalLedger<N> {
        self.engine.ledger()
    }

    pub fn records(&self) -> &[ReductionRecord] {
        self.engine.records()
    }

    pub fn skips(&self) -> u64 {
        self.skips
    }

    pub fn step<E>(
        &mut self,
        population: &Population<N>,
        signals: &CompactionSignals,
        evidence: &E,
        rng: &mut dyn RandomSource,
        check: &mut CandidateCheck<'_, N>,
    ) -> Option<ReductionReport<N>>
    where
        E: EvidenceSource<N> + ?Sized,
    {
        if self.engine.parameters.mode == CompactionMode::Off {
            return None;
        }
        let observation = self.window.observation(signals);
        let TriggerDecision::Compact(strength) = self.policy.decide(&observation) else {
            self.skips += 1;
            return None;
        };
        self.window.reset(signals);
        let request = ReductionRequest {
            interaction: signals.interactions,
            observation,
            strength,
        };
        let report = self
            .engine
            .reduce(&request, population, evidence, rng, check);
        self.policy.record_reduction(&report.feedback);
        Some(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::audit::Violation;
    use crate::compaction::test_support::{
        classifier, grounded_population, maximum_fitness_table, random_population, transition,
        ACTIONS, THETA_EXP, THETA_R,
    };
    use crate::compaction::trigger::{FeedbackParameters, Once, Periodic};
    use crate::support::rng::ChaChaRandomSource;

    struct CountingRandom {
        inner: ChaChaRandomSource,
        calls: usize,
    }

    impl CountingRandom {
        fn new(seed: u64) -> Self {
            Self {
                inner: ChaChaRandomSource::from_seed(seed),
                calls: 0,
            }
        }
    }

    impl RandomSource for CountingRandom {
        fn gen_bool(&mut self, probability: f64) -> bool {
            self.calls += 1;
            self.inner.gen_bool(probability)
        }
        fn gen_range(&mut self, bound: usize) -> usize {
            self.calls += 1;
            self.inner.gen_range(bound)
        }
        fn gen_unit(&mut self) -> f64 {
            self.calls += 1;
            self.inner.gen_unit()
        }
    }

    fn geometry() -> TaskGeometry {
        TaskGeometry {
            actions: ACTIONS,
            goal_attributes: 2..3,
        }
    }

    fn strength(tolerance: f64, maturity: u32, audits: u32) -> CompactionStrength {
        CompactionStrength {
            value_tolerance: tolerance,
            maturity_experience: maturity,
            idle_audits: audits,
        }
    }

    fn exact_strength() -> CompactionStrength {
        strength(0.0, THETA_EXP, 1)
    }

    fn once_now() -> CompactionPolicy {
        fixed_strength_policy(Once::new(0), exact_strength()).unwrap()
    }

    fn periodic(every: u64) -> CompactionPolicy {
        fixed_strength_policy(Periodic::new(every).unwrap(), exact_strength()).unwrap()
    }

    fn core_parameters(mode: CompactionMode, policy: CompactionPolicy) -> CompactionParameters {
        CompactionParameters::builder(THETA_R, policy)
            .mode(mode)
            .operator(Operator::MatureSubsumption { guarded: true })
            .audit(AuditPolicy::Observe(AuditThresholds::lossless()))
            .evidence_sample(6)
            .build()
            .unwrap()
    }

    fn signals(interactions: u64, population: &Population<3>) -> CompactionSignals {
        CompactionSignals {
            interactions,
            population_size: population.len(),
            match_formations: interactions * 10,
            classifier_tests: interactions * 100,
            ..CompactionSignals::default()
        }
    }

    fn debug(population: &Population<3>) -> String {
        format!("{:?}", population.classifiers())
    }

    fn redundant_pair() -> Population<3> {
        Population::from_classifiers(vec![
            classifier(b"10#", 0, b"##1", 0.5),
            classifier(b"1##", 0, b"##1", 0.95),
        ])
    }

    fn run(
        controller: &mut CompactionController<3>,
        population: &Population<3>,
        interactions: u64,
        evidence: &[Transition<3>],
        rng: &mut dyn RandomSource,
    ) -> Option<ReductionReport<3>> {
        controller.step(
            population,
            &signals(interactions, population),
            evidence,
            rng,
            &mut no_candidate_check,
        )
    }

    #[test]
    fn off_never_touches_the_population_the_policy_or_the_random_stream() {
        let evidence: Vec<_> = (0..20).map(|_| transition(b"100", 0, b"101")).collect();
        let mut controller =
            CompactionController::<3>::new(CompactionParameters::off(), geometry());
        let mut rng = CountingRandom::new(1);
        let population = redundant_pair();
        for interactions in [0, 10, 500] {
            assert!(run(&mut controller, &population, interactions, &evidence, &mut rng).is_none());
        }
        assert_eq!(rng.calls, 0);
        assert!(controller.records().is_empty());
        assert_eq!(controller.skips(), 0);
        assert_eq!(controller.policy().fires(), 0);
        assert_eq!(controller.mode(), CompactionMode::Off);
        assert_eq!(debug(&population), debug(&redundant_pair()));
    }

    #[test]
    fn shadow_reports_a_candidate_but_never_replaces_or_remembers_removals() {
        let mut controller = CompactionController::<3>::new(
            core_parameters(CompactionMode::Shadow, periodic(10)),
            geometry(),
        );
        let mut rng = CountingRandom::new(2);
        let population = redundant_pair();
        let before = debug(&population);
        let report = run(&mut controller, &population, 10, &[], &mut rng).unwrap();
        assert!(!report.applied);
        assert_eq!(report.candidate.as_ref().map(Population::len), Some(1));
        assert!(report.replacement().is_none());
        assert_eq!(debug(&population), before);
        let record = &controller.records()[0];
        assert_eq!(record.mode, CompactionMode::Shadow);
        assert!(record.accepted && !record.applied);
        assert_eq!((record.size_before, record.size_candidate, record.size_after), (2, 1, 2));
        assert_eq!(record.removals.subsumed, 1);
        assert!(controller.ledger().is_empty());
        assert!(record.restoration.is_none());
    }

    #[test]
    fn shadow_leaves_random_populations_identical() {
        for seed in 0..25 {
            let mut generator = ChaChaRandomSource::from_seed(seed);
            let population = random_population(&mut generator, 12);
            let before = debug(&population);
            let mut controller = CompactionController::<3>::new(
                core_parameters(CompactionMode::Shadow, once_now()),
                geometry(),
            );
            let mut rng = CountingRandom::new(seed);
            let report = run(&mut controller, &population, 1, &[], &mut rng);
            assert!(report.is_some_and(|report| !report.applied));
            assert_eq!(debug(&population), before);
        }
    }

    #[test]
    fn apply_with_exact_guard_keeps_every_maximum_fitness_bootstrap() {
        let mut compressed = 0;
        for seed in 0..60 {
            let mut generator = ChaChaRandomSource::from_seed(seed);
            let population = if seed % 2 == 0 {
                random_population(&mut generator, 14)
            } else {
                grounded_population(&mut generator, 14)
            };
            let table = maximum_fitness_table(&population);
            let mut controller = CompactionController::<3>::new(
                core_parameters(CompactionMode::Apply, once_now()),
                geometry(),
            );
            let mut rng = CountingRandom::new(seed);
            let report = run(&mut controller, &population, 1, &[], &mut rng).unwrap();
            let record = controller.records()[0].clone();
            assert!(record.accepted);
            let reduced = report.replacement();
            compressed += usize::from(reduced.is_some());
            let after = reduced.as_ref().unwrap_or(&population);
            assert_eq!(maximum_fitness_table(after), table, "seed {seed}");
            assert_eq!(record.size_after, after.len());
            assert_eq!(record.size_before - record.removals.total(), record.size_after);
        }
        assert!(compressed > 0);
    }

    #[test]
    fn a_rejected_audit_rolls_the_pass_back() {
        let population = Population::from_classifiers(vec![
            classifier(b"1##", 0, b"##1", 0.93),
            classifier(b"0##", 1, b"###", 0.2),
        ]);
        let evidence = [transition(b"100", 0, b"101")];
        let feedback = FeedbackParameters::new(0, 1000, 1000, 0, 0, 0).unwrap();
        let parameters = CompactionParameters::builder(
            THETA_R,
            fixed_strength_policy(Once::new(0), exact_strength())
                .unwrap()
                .with_feedback(feedback),
        )
        .operator(Operator::DynamicThresholds(DynamicThresholds::new(0, 0, 0.95)))
        .audit(AuditPolicy::Enforce(AuditThresholds::lossless()))
        .evidence_sample(4)
        .build()
        .unwrap();
        let mut controller = CompactionController::<3>::new(parameters, geometry());
        let mut rng = CountingRandom::new(3);
        let report = run(&mut controller, &population, 5, &evidence, &mut rng).unwrap();
        assert!(!report.applied);
        assert!(report.feedback.rejected_by_guard);
        assert!(report.replacement().is_none());
        let record = &controller.records()[0];
        assert!(!record.accepted && !record.applied);
        assert!(record
            .verdict
            .as_ref()
            .unwrap()
            .violations
            .contains(&Violation::CoverageLoss));
        assert_eq!(record.size_candidate, 0);
        assert_eq!(record.size_after, 2);
        assert!(controller.ledger().is_empty());
        assert_eq!(controller.policy().last_yield_permille(), Some(0));
    }

    #[test]
    fn an_external_check_can_veto_an_otherwise_accepted_pass() {
        let mut controller = CompactionController::<3>::new(
            core_parameters(CompactionMode::Apply, once_now()),
            geometry(),
        );
        let mut rng = CountingRandom::new(4);
        let population = redundant_pair();
        let none: [Transition<3>; 0] = [];
        let mut veto = |_: &Population<3>, _: &Population<3>, _: &[Transition<3>]| CheckResult {
            vetoes: true,
            work: 7,
        };
        let report = controller
            .step(&population, &signals(1, &population), &none[..], &mut rng, &mut veto)
            .unwrap();
        assert!(!report.applied);
        let record = &controller.records()[0];
        assert!(record.check_vetoed && !record.accepted);
        assert_eq!(record.check_work, 7);
        assert_eq!(record.charged_work, record.operator_work + 7);
    }

    #[test]
    fn applied_removals_enter_the_ledger_and_restoration_steers_the_strength() {
        let feedback = FeedbackParameters::new(0, 1000, 100, 0, 0, 0).unwrap();
        let bounds = StrengthBounds::new(
            strength(0.0, 40, 6),
            strength(0.5, 4, 1),
            StrengthStep::new(0.125, 4, 1).unwrap(),
        )
        .unwrap();
        let policy = CompactionPolicy::new(
            Periodic::new(10).unwrap(),
            bounds,
            strength(0.25, 12, 3),
        )
        .with_feedback(feedback);
        let parameters = CompactionParameters::builder(THETA_R, policy)
            .operator(Operator::MatureSubsumption { guarded: true })
            .build()
            .unwrap();
        let mut controller = CompactionController::<3>::new(parameters, geometry());
        let mut rng = CountingRandom::new(5);

        let first = redundant_pair();
        let reduced = run(&mut controller, &first, 10, &[], &mut rng)
            .unwrap()
            .replacement()
            .unwrap();
        assert_eq!(reduced.len(), 1);
        assert_eq!(controller.ledger().len(), 1);
        assert!(controller.records()[0].restoration.is_none());
        assert_eq!(controller.policy().strength(), strength(0.375, 8, 2));

        let regrown = redundant_pair();
        assert!(run(&mut controller, &regrown, 15, &[], &mut rng).is_none());
        let second = run(&mut controller, &regrown, 20, &[], &mut rng).unwrap();
        let record = &controller.records()[1];
        assert_eq!(
            record.restoration,
            Some(Restoration {
                removed: 1,
                restored: 1
            })
        );
        assert_eq!(second.feedback.restored_permille, Some(1000));
        assert_eq!(controller.policy().strength(), strength(0.25, 12, 3));
        assert_eq!(record.strength, strength(0.375, 8, 2));
        assert_eq!(controller.skips(), 1);
    }

    #[test]
    fn the_signal_window_reports_work_since_the_previous_fire() {
        let mut controller = CompactionController::<3>::new(
            core_parameters(CompactionMode::Apply, periodic(10)),
            geometry(),
        );
        let mut rng = CountingRandom::new(6);
        let population = redundant_pair();
        run(&mut controller, &population, 10, &[], &mut rng).unwrap();
        run(&mut controller, &population, 20, &[], &mut rng).unwrap();
        let observed: Vec<_> = controller
            .records()
            .iter()
            .map(|record| {
                (
                    record.observation.match_formations_since_reduction,
                    record.observation.classifier_tests_since_reduction,
                )
            })
            .collect();
        assert_eq!(observed, vec![(100, 1000), (100, 1000)]);
    }

    #[test]
    fn idle_pruning_waits_for_its_required_audits_and_counts_the_work() {
        let population = Population::from_classifiers(vec![
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"10#", 0, b"##1", 0.93),
        ]);
        let evidence = [transition(b"100", 0, b"101")];
        let parameters = CompactionParameters::builder(
            THETA_R,
            fixed_strength_policy(Periodic::new(1).unwrap(), strength(0.0, 0, 2)).unwrap(),
        )
        .operator(Operator::IdlePruning { min_experience: 0 })
        .audit(AuditPolicy::Enforce(AuditThresholds::lossless()))
        .evidence_sample(4)
        .build()
        .unwrap();
        let mut controller = CompactionController::<3>::new(parameters, geometry());
        let mut rng = CountingRandom::new(7);
        let first = run(&mut controller, &population, 1, &evidence, &mut rng).unwrap();
        assert!(first.candidate.is_none());
        assert_eq!(controller.tracker().len(), 1);
        let second = run(&mut controller, &population, 2, &evidence, &mut rng).unwrap();
        assert!(second.applied);
        let record = &controller.records()[1];
        assert_eq!(record.removals.idle, 1);
        assert!(record.operator_work > 0 && record.audit_work > 0);
        assert_eq!(record.charged_work, record.operator_work + record.audit_work);
        assert!(record.census_before.is_some() && record.census_candidate.is_some());
        assert_eq!(record.stages.len(), 1);
        assert_eq!(record.stages[0].operator, OperatorKind::IdlePruning);
    }

    #[test]
    fn evidence_is_subsampled_from_the_supplied_stream_only_when_a_pass_runs() {
        let pool: Vec<_> = (0..40).map(|_| transition(b"100", 0, b"101")).collect();
        let mut controller = CompactionController::<3>::new(
            core_parameters(CompactionMode::Apply, periodic(10)),
            geometry(),
        );
        let mut rng = CountingRandom::new(8);
        let population = redundant_pair();
        run(&mut controller, &population, 5, &pool, &mut rng);
        assert_eq!(rng.calls, 0);
        run(&mut controller, &population, 10, &pool, &mut rng);
        assert_eq!(rng.calls, 6);
        assert_eq!(controller.records()[0].evidence_size, 6);
    }

    #[test]
    fn the_same_inputs_give_the_same_controller_state() {
        let pool: Vec<_> = (0..30)
            .map(|index| {
                if index % 2 == 0 {
                    transition(b"100", 0, b"101")
                } else {
                    transition(b"011", 1, b"010")
                }
            })
            .collect();
        let play = || {
            let mut controller = CompactionController::<3>::new(
                core_parameters(CompactionMode::Apply, periodic(5)),
                geometry(),
            );
            let mut rng = ChaChaRandomSource::from_seed_and_stream(42, COMPACTION_STREAM);
            for seed in 0..8 {
                let mut generator = ChaChaRandomSource::from_seed(seed);
                let population = random_population(&mut generator, 10);
                run(&mut controller, &population, 5 * (seed + 1), &pool, &mut rng);
            }
            (format!("{controller:?}"), rng.capture_state())
        };
        assert_eq!(play(), play());
    }

    #[test]
    fn a_second_pipeline_stage_sees_the_population_the_first_one_left() {
        let population = Population::from_classifiers(vec![
            classifier(b"10#", 0, b"##1", 0.5),
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"0##", 1, b"###", 0.2),
        ]);
        let parameters = CompactionParameters::builder(THETA_R, once_now())
            .operator(Operator::MatureSubsumption { guarded: true })
            .operator(Operator::DynamicThresholds(DynamicThresholds::new(0, 0, 0.3)))
            .census(false)
            .build()
            .unwrap();
        let mut controller = CompactionController::<3>::new(parameters, geometry());
        let mut rng = CountingRandom::new(9);
        let report = run(&mut controller, &population, 1, &[], &mut rng).unwrap();
        let record = &controller.records()[0];
        assert_eq!(record.removals.subsumed, 1);
        assert_eq!(record.removals.low_quality, 1);
        assert_eq!(record.stages.len(), 2);
        assert_eq!(report.replacement().unwrap().len(), 1);
        assert_eq!(controller.ledger().len(), 2);
        assert!(record.census_before.is_none());
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        let builder = |theta_r| CompactionParameters::builder(theta_r, once_now());
        assert_eq!(
            builder(THETA_R).build().unwrap_err(),
            CompactionParameterError::NoOperators
        );
        let subsume = Operator::MatureSubsumption { guarded: true };
        assert_eq!(
            builder(THETA_R)
                .operator(subsume)
                .operator(subsume)
                .build()
                .unwrap_err(),
            CompactionParameterError::DuplicateOperator(OperatorKind::MatureSubsumption)
        );
        assert_eq!(
            builder(0.0).operator(subsume).build().unwrap_err(),
            CompactionParameterError::ReliabilityOutOfRange
        );
        assert_eq!(
            builder(THETA_R)
                .operator(Operator::IdlePruning { min_experience: 0 })
                .build()
                .unwrap_err(),
            CompactionParameterError::EvidenceRequired
        );
        assert_eq!(
            builder(THETA_R)
                .operator(subsume)
                .audit(AuditPolicy::Enforce(AuditThresholds::new(f64::NAN, 0.0, 0.0, 0.0)))
                .evidence_sample(3)
                .build()
                .unwrap_err(),
            CompactionParameterError::InvalidAuditThresholds
        );
        assert!(builder(THETA_R)
            .operator(subsume)
            .census(false)
            .build()
            .is_ok());
        assert_eq!(CompactionParameters::off().mode(), CompactionMode::Off);
    }
}
