use acs2_core::compaction::{
    CheckResult, CompactionController, CompactionMode, CompactionParameters, CompactionPolicy,
    CompactionSignals, EvidenceSource, ReductionEngine, ReductionFeedback, ReductionRecord,
    ReductionRequest, SignalWindow, TaskGeometry, TriggerDecision, COMPACTION_STREAM,
};
use acs2_core::learning::replay::ReplaySample;
use acs2_core::ports::goal::GoalLayout;
use acs2_core::representation::population::Population;
use acs2_core::support::config::Configuration;
use acs2_core::support::knowledge::Transition;
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_trajectory::TrajectoryStore;

use super::priority::{acceptance, action_value_uncounted, population_variance};
use super::{Agent, ConsistencyConfiguration};
use crate::episodic_replay::{EpisodicConfiguration, ReplayEnvironment, UpdateCounts};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompactionClock {
    PerHead,
    Synchronous,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AcceptanceLimits {
    pub max_mean_acceptance_shift: f64,
    pub max_mean_variance_shift: f64,
}

impl AcceptanceLimits {
    pub fn new(max_mean_acceptance_shift: f64, max_mean_variance_shift: f64) -> Self {
        assert!(
            max_mean_acceptance_shift.is_finite()
                && max_mean_acceptance_shift >= 0.0
                && max_mean_variance_shift.is_finite()
                && max_mean_variance_shift >= 0.0,
            "acceptance limits must be finite and non-negative"
        );
        Self {
            max_mean_acceptance_shift,
            max_mean_variance_shift,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DisagreementProbe {
    pub every_episodes: u64,
    pub sample: usize,
}

impl DisagreementProbe {
    pub fn new(every_episodes: u64, sample: usize) -> Self {
        assert!(
            every_episodes > 0 && sample > 0,
            "a disagreement probe needs a positive period and sample"
        );
        Self {
            every_episodes,
            sample,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactingConfiguration {
    clock: CompactionClock,
    compaction: CompactionParameters,
    acceptance_limits: Option<AcceptanceLimits>,
    disagreement_probe: Option<DisagreementProbe>,
}

impl CompactingConfiguration {
    pub fn new(clock: CompactionClock, compaction: CompactionParameters) -> Self {
        Self {
            clock,
            compaction,
            acceptance_limits: None,
            disagreement_probe: None,
        }
    }

    pub fn off() -> Self {
        Self::new(CompactionClock::PerHead, CompactionParameters::off())
    }

    pub fn with_acceptance_limits(mut self, limits: AcceptanceLimits) -> Self {
        self.acceptance_limits = Some(limits);
        self
    }

    pub fn with_disagreement_probe(mut self, probe: DisagreementProbe) -> Self {
        self.disagreement_probe = Some(probe);
        self
    }

    pub fn clock(&self) -> CompactionClock {
        self.clock
    }

    pub fn compaction(&self) -> &CompactionParameters {
        &self.compaction
    }

    pub fn acceptance_limits(&self) -> Option<AcceptanceLimits> {
        self.acceptance_limits
    }

    pub fn disagreement_probe(&self) -> Option<DisagreementProbe> {
        self.disagreement_probe
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct HeadCost {
    pub match_formations: u64,
    pub classifier_tests: u64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AcceptanceSummary {
    pub mean_variance: f64,
    pub mean_acceptance: f64,
}

#[derive(Clone, PartialEq, Debug)]
pub struct AcceptanceAuditRecord {
    pub head: usize,
    pub sequence: u64,
    pub interaction: u64,
    pub samples: usize,
    pub before: AcceptanceSummary,
    pub after: AcceptanceSummary,
    pub vetoed: bool,
    pub work: u64,
}

#[derive(Clone, PartialEq, Debug)]
pub struct DisagreementRecord {
    pub episodes: u64,
    pub interactions: u64,
    pub samples: usize,
    pub summary: AcceptanceSummary,
    pub head_sizes: Vec<usize>,
    pub work: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct DiagnosticLog {
    acceptance: Vec<AcceptanceAuditRecord>,
    probes: Vec<DisagreementRecord>,
    acceptance_work: u64,
    probe_work: u64,
}

#[derive(Clone, Debug, PartialEq)]
struct SynchronousController<const M: usize> {
    policy: CompactionPolicy,
    window: SignalWindow,
    engines: Vec<ReductionEngine<M>>,
    skips: u64,
}

#[derive(Clone, Debug, PartialEq)]
enum Clock<const M: usize> {
    PerHead(Vec<CompactionController<M>>),
    Synchronous(SynchronousController<M>),
}

struct BoundaryMark {
    population_size: usize,
    counts: UpdateCounts,
    value_queries: u64,
}

struct StoredEvidence<'a, const S: usize, const G: usize> {
    store: &'a TrajectoryStore<S, G>,
}

impl<const S: usize, const G: usize> StoredEvidence<'_, S, G> {
    fn collect<const M: usize>(&self, indices: &[usize]) -> Vec<Transition<M>> {
        let mut transitions = Vec::with_capacity(indices.len());
        let mut pending = indices.iter().copied().peekable();
        let mut offset = 0usize;
        for episode in self.store.episodes() {
            let end = offset + episode.len();
            while let Some(index) = pending.next_if(|&index| index < end) {
                let local = index - offset;
                let desired = episode.start().desired;
                transitions.push(Transition::new(
                    GoalLayout::<S, G, M>::join(episode.observation(local), &desired),
                    episode.steps()[local].action,
                    GoalLayout::<S, G, M>::join(episode.observation(local + 1), &desired),
                ));
            }
            offset = end;
        }
        transitions
    }
}

impl<const S: usize, const G: usize, const M: usize> EvidenceSource<M> for StoredEvidence<'_, S, G> {
    fn sample(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<Transition<M>> {
        let indices = acs2_core::compaction::subsample_indices(self.store.len(), size, rng);
        self.collect::<M>(&indices)
    }
}

struct AcceptanceAudit<'a, const M: usize> {
    heads: Vec<&'a Population<M>>,
    temperature: u32,
    maximum: f64,
    limits: Option<AcceptanceLimits>,
}

struct AuditContext {
    head: usize,
    sequence: u64,
    interaction: u64,
}

impl<'a, const M: usize> AcceptanceAudit<'a, M> {
    fn of<const S: usize, const G: usize, R, P, A>(
        agent: &'a Agent<S, G, M, R, P, A>,
        limits: Option<AcceptanceLimits>,
    ) -> Self
    where
        R: RandomSource,
        P: RandomSource,
        A: RandomSource,
    {
        Self {
            heads: agent.heads().iter().map(|head| head.population()).collect(),
            temperature: agent.parameters().temperature,
            maximum: agent.parameters().variance_maximum(),
            limits,
        }
    }

    fn summarize(
        &self,
        samples: &[ReplaySample<M>],
        changed_head: usize,
        substitute: Option<&Population<M>>,
    ) -> (AcceptanceSummary, u64) {
        let populations: Vec<&Population<M>> = self
            .heads
            .iter()
            .enumerate()
            .map(|(index, population)| match substitute {
                Some(candidate) if index == changed_head => candidate,
                _ => *population,
            })
            .collect();
        let tested: u64 = populations.iter().map(|population| population.len() as u64).sum();
        let mut variance_total = 0.0;
        let mut acceptance_total = 0.0;
        for sample in samples {
            let variance = if populations.len() == 1 {
                0.0
            } else {
                let values: Vec<f64> = populations
                    .iter()
                    .map(|population| action_value_uncounted(population, sample))
                    .collect();
                population_variance(&values).min(self.maximum)
            };
            variance_total += variance;
            acceptance_total += acceptance(variance, self.maximum, self.temperature);
        }
        let count = samples.len().max(1) as f64;
        (
            AcceptanceSummary {
                mean_variance: variance_total / count,
                mean_acceptance: acceptance_total / count,
            },
            samples.len() as u64 * tested,
        )
    }

    fn check(
        &self,
        log: &mut DiagnosticLog,
        context: &AuditContext,
        candidate: &Population<M>,
        evidence: &[Transition<M>],
    ) -> CheckResult {
        let samples = replay_samples(evidence);
        let (before, before_work) = self.summarize(&samples, context.head, None);
        let (after, after_work) = self.summarize(&samples, context.head, Some(candidate));
        let work = before_work + after_work;
        let vetoed = self.limits.is_some_and(|limits| {
            (after.mean_acceptance - before.mean_acceptance).abs()
                > limits.max_mean_acceptance_shift
                || (after.mean_variance - before.mean_variance).abs()
                    > limits.max_mean_variance_shift
        });
        log.acceptance_work += work;
        log.acceptance.push(AcceptanceAuditRecord {
            head: context.head,
            sequence: context.sequence,
            interaction: context.interaction,
            samples: samples.len(),
            before,
            after,
            vetoed,
            work,
        });
        CheckResult {
            vetoes: vetoed,
            work: if self.limits.is_some() { work } else { 0 },
        }
    }
}

fn replay_samples<const M: usize>(evidence: &[Transition<M>]) -> Vec<ReplaySample<M>> {
    evidence
        .iter()
        .map(|transition| ReplaySample {
            state: transition.p0,
            action: transition.action,
            reward: 0.0,
            next_state: transition.p1,
            done: false,
        })
        .collect()
}

pub struct CompactingAgent<const S: usize, const G: usize, const M: usize, R, P, A, C> {
    agent: Agent<S, G, M, R, P, A>,
    configuration: CompactingConfiguration,
    clock: Clock<M>,
    costs: Vec<HeadCost>,
    rng: C,
    log: DiagnosticLog,
}

impl<const S: usize, const G: usize, const M: usize>
    CompactingAgent<
        S,
        G,
        M,
        ChaChaRandomSource,
        ChaChaRandomSource,
        ChaChaRandomSource,
        ChaChaRandomSource,
    >
{
    pub fn seeded(
        config: Configuration,
        replay: EpisodicConfiguration,
        parameters: ConsistencyConfiguration,
        compacting: CompactingConfiguration,
        seed: u64,
        agent_stream: u64,
    ) -> Self {
        Self::new(
            Agent::seeded(config, replay, parameters, seed, agent_stream),
            compacting,
            ChaChaRandomSource::from_seed_and_stream(seed, COMPACTION_STREAM),
        )
    }
}

impl<
        const S: usize,
        const G: usize,
        const M: usize,
        R: RandomSource,
        P: RandomSource,
        A: RandomSource,
        C: RandomSource,
    > CompactingAgent<S, G, M, R, P, A, C>
{
    pub fn new(
        agent: Agent<S, G, M, R, P, A>,
        configuration: CompactingConfiguration,
        rng: C,
    ) -> Self {
        let geometry = TaskGeometry {
            actions: agent.config().number_of_possible_actions,
            goal_attributes: GoalLayout::<S, G, M>::goal_positions(),
        };
        let heads = agent.heads().len();
        let parameters = configuration.compaction.clone();
        let clock = match configuration.clock {
            CompactionClock::PerHead => Clock::PerHead(
                (0..heads)
                    .map(|_| CompactionController::new(parameters.clone(), geometry.clone()))
                    .collect(),
            ),
            CompactionClock::Synchronous => Clock::Synchronous(SynchronousController {
                policy: parameters.policy().clone(),
                window: SignalWindow::default(),
                engines: (0..heads)
                    .map(|_| ReductionEngine::new(parameters.clone(), geometry.clone()))
                    .collect(),
                skips: 0,
            }),
        };
        Self {
            agent,
            configuration,
            clock,
            costs: vec![HeadCost::default(); heads],
            rng,
            log: DiagnosticLog::default(),
        }
    }

    pub fn agent(&self) -> &Agent<S, G, M, R, P, A> {
        &self.agent
    }

    pub fn configuration(&self) -> &CompactingConfiguration {
        &self.configuration
    }

    pub fn mode(&self) -> CompactionMode {
        self.configuration.compaction.mode()
    }

    pub fn head_cost(&self, head: usize) -> HeadCost {
        self.costs[head]
    }

    pub fn reduction_records(&self, head: usize) -> &[ReductionRecord] {
        match &self.clock {
            Clock::PerHead(controllers) => controllers[head].records(),
            Clock::Synchronous(synchronous) => synchronous.engines[head].records(),
        }
    }

    pub fn head_controllers(&self) -> Option<&[CompactionController<M>]> {
        match &self.clock {
            Clock::PerHead(controllers) => Some(controllers),
            Clock::Synchronous(_) => None,
        }
    }

    pub fn synchronous_policy(&self) -> Option<&CompactionPolicy> {
        match &self.clock {
            Clock::PerHead(_) => None,
            Clock::Synchronous(synchronous) => Some(&synchronous.policy),
        }
    }

    pub fn acceptance_audits(&self) -> &[AcceptanceAuditRecord] {
        &self.log.acceptance
    }

    pub fn disagreement_records(&self) -> &[DisagreementRecord] {
        &self.log.probes
    }

    pub fn acceptance_work(&self) -> u64 {
        self.log.acceptance_work
    }

    pub fn probe_work(&self) -> u64 {
        self.log.probe_work
    }

    pub fn compaction_random_source(&self) -> &C {
        &self.rng
    }

    pub fn snapshot(&self) -> String {
        format!(
            "{}|{:?}|{:?}|{:?}|{:?}|{:?}",
            self.agent.snapshot(),
            self.configuration,
            self.clock,
            self.costs,
            self.rng.capture_state(),
            self.log
        )
    }

    pub fn train_episode(&mut self, env: &mut impl ReplayEnvironment<S, G, M>, time: u64) {
        self.train_episode_observed(env, time, &mut |_, _| {});
    }

    pub fn train_episode_observed(
        &mut self,
        env: &mut impl ReplayEnvironment<S, G, M>,
        time: u64,
        observe: &mut impl FnMut(f64, bool),
    ) {
        if self.mode() == CompactionMode::Off {
            self.agent.train_episode_observed(env, time, observe);
        } else {
            let playing = self.agent.playing_head();
            let mark = BoundaryMark {
                population_size: self.agent.heads()[playing].population().len(),
                counts: self.agent.heads()[playing].counts(),
                value_queries: self.agent.priority_counts().value_queries,
            };
            self.agent.train_episode_observed(env, time, observe);
            self.account_matching_cost(playing, &mark);
            match self.configuration.clock {
                CompactionClock::PerHead => self.compact_per_head(),
                CompactionClock::Synchronous => self.compact_synchronously(),
            }
        }
        self.probe_disagreement();
    }

    fn account_matching_cost(&mut self, playing: usize, mark: &BoundaryMark) {
        let heads = self.agent.heads();
        let counts = heads[playing].counts();
        let environment = counts.environment_steps - mark.counts.environment_steps;
        let learning = 3 * (counts.full - mark.counts.full)
            + 2 * (counts.values_only - mark.counts.values_only);
        let ended = heads[playing].population().len() as u64;
        let started = mark.population_size as u64;
        let queries = (self.agent.priority_counts().value_queries - mark.value_queries)
            / heads.len() as u64;
        for (index, cost) in self.costs.iter_mut().enumerate() {
            let tested_at_draw = if index == playing {
                started
            } else {
                heads[index].population().len() as u64
            };
            cost.match_formations += queries;
            cost.classifier_tests += queries * tested_at_draw;
        }
        let cost = &mut self.costs[playing];
        cost.match_formations += environment + learning;
        cost.classifier_tests += environment * started + learning * ((started + ended) / 2);
    }

    fn compact_per_head(&mut self) {
        let Clock::PerHead(controllers) = &mut self.clock else {
            return;
        };
        let audit = AcceptanceAudit::of(&self.agent, self.configuration.acceptance_limits);
        let evidence = StoredEvidence {
            store: self.agent.store(),
        };
        let mut replacements = Vec::new();
        for (head, controller) in controllers.iter_mut().enumerate() {
            let state = &self.agent.heads()[head];
            let signals = CompactionSignals {
                interactions: state.counts().environment_steps,
                population_size: state.population().len(),
                match_formations: self.costs[head].match_formations,
                classifier_tests: self.costs[head].classifier_tests,
                rules_created: None,
                coverage_permille: None,
            };
            let context = AuditContext {
                head,
                sequence: controller.records().len() as u64,
                interaction: signals.interactions,
            };
            let log = &mut self.log;
            let report = controller.step(
                state.population(),
                &signals,
                &evidence,
                &mut self.rng,
                &mut |_, candidate, sample| audit.check(log, &context, candidate, sample),
            );
            if let Some(population) = report.and_then(|report| report.replacement()) {
                replacements.push((head, population));
            }
        }
        for (head, population) in replacements {
            self.agent.replace_population(head, population);
        }
    }

    fn compact_synchronously(&mut self) {
        let Clock::Synchronous(synchronous) = &mut self.clock else {
            return;
        };
        let heads = self.agent.heads();
        let signals = CompactionSignals {
            interactions: heads.iter().map(|head| head.counts().environment_steps).sum(),
            population_size: heads.iter().map(|head| head.population().len()).sum(),
            match_formations: self.costs.iter().map(|cost| cost.match_formations).sum(),
            classifier_tests: self.costs.iter().map(|cost| cost.classifier_tests).sum(),
            rules_created: None,
            coverage_permille: None,
        };
        let observation = synchronous.window.observation(&signals);
        let TriggerDecision::Compact(strength) = synchronous.policy.decide(&observation) else {
            synchronous.skips += 1;
            return;
        };
        synchronous.window.reset(&signals);
        let request = ReductionRequest {
            interaction: signals.interactions,
            observation,
            strength,
        };
        let audit = AcceptanceAudit::of(&self.agent, self.configuration.acceptance_limits);
        let evidence = StoredEvidence {
            store: self.agent.store(),
        };
        let mut reports = Vec::with_capacity(heads.len());
        for (head, engine) in synchronous.engines.iter_mut().enumerate() {
            let context = AuditContext {
                head,
                sequence: engine.records().len() as u64,
                interaction: signals.interactions,
            };
            let log = &mut self.log;
            reports.push(engine.reduce(
                &request,
                heads[head].population(),
                &evidence,
                &mut self.rng,
                &mut |_, candidate, sample| audit.check(log, &context, candidate, sample),
            ));
        }
        let feedback = ReductionFeedback {
            population_before: reports.iter().map(|r| r.feedback.population_before).sum(),
            population_after: reports.iter().map(|r| r.feedback.population_after).sum(),
            work_classifier_tests: reports
                .iter()
                .map(|r| r.feedback.work_classifier_tests)
                .sum(),
            rejected_by_guard: reports.iter().any(|r| r.feedback.rejected_by_guard),
            restored_permille: reports
                .iter()
                .filter_map(|r| r.feedback.restored_permille)
                .max(),
        };
        synchronous.policy.record_reduction(&feedback);
        let replacements: Vec<_> = reports
            .into_iter()
            .enumerate()
            .filter_map(|(head, report)| report.replacement().map(|population| (head, population)))
            .collect();
        for (head, population) in replacements {
            self.agent.replace_population(head, population);
        }
    }

    fn probe_disagreement(&mut self) {
        let Some(probe) = self.configuration.disagreement_probe else {
            return;
        };
        let heads = self.agent.heads();
        let episodes: u64 = heads.iter().map(|head| head.episodes()).sum();
        if episodes == 0 || episodes % probe.every_episodes != 0 {
            return;
        }
        let store = self.agent.store();
        let indices = acs2_core::compaction::strided_indices(store.len(), probe.sample);
        let evidence = StoredEvidence { store }.collect::<M>(&indices);
        let samples = replay_samples(&evidence);
        let audit = AcceptanceAudit::of(&self.agent, None);
        let (summary, work) = audit.summarize(&samples, 0, None);
        self.log.probe_work += work;
        self.log.probes.push(DisagreementRecord {
            episodes,
            interactions: heads
                .iter()
                .map(|head| head.counts().environment_steps)
                .sum(),
            samples: samples.len(),
            summary,
            head_sizes: heads.iter().map(|head| head.population().len()).collect(),
            work,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acs2_core::compaction::trigger::{CostBenefit, Once, Periodic};
    use acs2_core::compaction::{
        fixed_strength_policy, AuditPolicy, AuditThresholds, CompactionStrength, DynamicThresholds,
        Operator,
    };
    use acs2_core::ports::environment::{Environment, StepOutcome};
    use acs2_core::ports::goal::{ExactMatch, Goal, GoalOutcome, GoalStart, GoalStep};
    use acs2_core::representation::classifier::Classifier;
    use acs2_core::representation::perception::Perception;
    use acs2_core::representation::symbol::Symbol;
    use acs2_core::support::measurement::{
        read_match_counters, start_match_counting, stop_match_counting,
    };
    use acs2_trajectory::{Admissibility, GoalEvaluator};

    use crate::acs2her::HindsightConfiguration;
    use crate::episodic_replay::CopyUpdate;
    use crate::acs2vcp::priority::action_value;

    type Plain = Agent<1, 1, 2, ChaChaRandomSource, ChaChaRandomSource, ChaChaRandomSource>;
    type Compacting = CompactingAgent<
        1,
        1,
        2,
        ChaChaRandomSource,
        ChaChaRandomSource,
        ChaChaRandomSource,
        ChaChaRandomSource,
    >;

    fn perception(values: [u8; 2]) -> Perception<2> {
        Perception {
            symbols: values.map(Symbol::Token),
        }
    }

    struct CycleEnvironment {
        state: u8,
        steps: usize,
        length: usize,
    }

    impl CycleEnvironment {
        fn new() -> Self {
            Self {
                state: 0,
                steps: 0,
                length: 3,
            }
        }
    }

    impl Environment<2> for CycleEnvironment {
        fn reset(&mut self) -> Perception<2> {
            self.state = 0;
            self.steps = 0;
            perception([0, 9])
        }
        fn step(&mut self, action: usize) -> StepOutcome<2> {
            if action == 1 {
                self.state = (self.state + 1) % 3;
            }
            self.steps += 1;
            StepOutcome {
                observation: perception([self.state, 9]),
                reward: 0.0,
                terminated: false,
                truncated: self.steps == self.length,
                info: (),
            }
        }
    }

    impl GoalEvaluator<1, 1> for CycleEnvironment {
        fn outcome(&self, step: &GoalStep<1, 1>, goal: &Goal<1>) -> GoalOutcome {
            step.outcome(
                &ExactMatch {
                    reward_on_reach: 1000.0,
                },
                goal,
            )
        }
        fn is_reached(&self, achieved: &Goal<1>, desired: &Goal<1>) -> bool {
            achieved == desired
        }
    }

    impl ReplayEnvironment<1, 1, 2> for CycleEnvironment {
        fn episode_start(&self) -> GoalStart<1, 1> {
            GoalStart {
                observation: Perception {
                    symbols: [Symbol::Token(0)],
                },
                achieved: Goal::new([Symbol::Token(0)]),
                desired: Goal::new([Symbol::Token(9)]),
            }
        }
        fn last_step(&self) -> GoalStep<1, 1> {
            GoalStep {
                observation: Perception {
                    symbols: [Symbol::Token(self.state)],
                },
                achieved: Goal::new([Symbol::Token(self.state)]),
                terminal_state: false,
                time_limit_reached: self.steps == self.length,
            }
        }
    }

    fn learning_configuration() -> Configuration {
        Configuration {
            number_of_possible_actions: 2,
            epsilon: 0.5,
            do_ga: false,
            ..Configuration::default_protocol()
        }
    }

    fn consistency(heads: usize) -> ConsistencyConfiguration {
        ConsistencyConfiguration {
            heads,
            temperature: 4,
            reward_on_reach: 1000.0,
            hindsight: HindsightConfiguration {
                copy_update: CopyUpdate::Full,
                admissibility: Admissibility::EveryTransition,
                relabeled_proportion: 2.0 / 3.0,
            },
        }
    }

    fn replay() -> EpisodicConfiguration {
        EpisodicConfiguration {
            capacity: 100,
            warmup: 1,
            updates_per_step: 3,
            truncation: acs2_core::trial::TruncationMode::Bootstrap,
        }
    }

    fn redundant_head(config: &Configuration) -> Population<2> {
        let rule = |condition: [Symbol; 2], value: f64| {
            let mut classifier = Classifier::general(Some(0), config);
            classifier.condition.symbols = condition;
            classifier.q = 1.0;
            classifier.r = value;
            classifier.exp = 30;
            classifier
        };
        Population::from_classifiers(vec![
            rule([Symbol::Token(0), Symbol::Token(9)], 50.0),
            rule([Symbol::Wildcard, Symbol::Token(9)], 100.0),
        ])
    }

    fn plain(heads: usize) -> Plain {
        let config = learning_configuration();
        let populations = (0..heads).map(|_| redundant_head(&config)).collect();
        Plain::seeded(config, replay(), consistency(heads), 42, 1).with_populations(populations)
    }

    fn compacting(heads: usize, configuration: CompactingConfiguration) -> Compacting {
        Compacting::new(
            plain(heads),
            configuration,
            ChaChaRandomSource::from_seed_and_stream(42, COMPACTION_STREAM),
        )
    }

    fn exact_strength() -> CompactionStrength {
        CompactionStrength {
            value_tolerance: 0.0,
            maturity_experience: 20,
            idle_audits: 1,
        }
    }

    fn core_parameters(
        mode: CompactionMode,
        policy: CompactionPolicy,
    ) -> CompactionParameters {
        CompactionParameters::builder(0.3, policy)
            .mode(mode)
            .operator(Operator::MatureSubsumption { guarded: true })
            .audit(AuditPolicy::Enforce(AuditThresholds::lossless()))
            .evidence_sample(8)
            .build()
            .unwrap()
    }

    fn periodic_policy(every: u64) -> CompactionPolicy {
        fixed_strength_policy(Periodic::new(every).unwrap(), exact_strength()).unwrap()
    }

    fn configuration(
        clock: CompactionClock,
        mode: CompactionMode,
        every: u64,
    ) -> CompactingConfiguration {
        CompactingConfiguration::new(clock, core_parameters(mode, periodic_policy(every)))
    }

    fn train(agent: &mut Compacting, episodes: u64) {
        let mut env = CycleEnvironment::new();
        for episode in 0..episodes {
            agent.train_episode(&mut env, episode);
        }
    }

    fn train_plain(agent: &mut Plain, episodes: u64) {
        let mut env = CycleEnvironment::new();
        for episode in 0..episodes {
            agent.train_episode(&mut env, episode);
        }
    }

    #[test]
    fn off_gives_the_snapshot_of_the_plain_agent() {
        for heads in [1, 2, 4] {
            let mut wrapped = compacting(heads, CompactingConfiguration::off());
            let mut reference = plain(heads);
            train(&mut wrapped, 9);
            train_plain(&mut reference, 9);
            assert_eq!(wrapped.agent().snapshot(), reference.snapshot());
            assert!((0..heads).all(|head| wrapped.reduction_records(head).is_empty()));
            assert_eq!(wrapped.compaction_random_source().capture_state().unwrap().word_pos, 0);
            assert_eq!(wrapped.head_cost(0), HeadCost::default());
        }
    }

    #[test]
    fn shadow_follows_the_plain_learning_trajectory_and_keeps_the_match_identity() {
        for clock in [CompactionClock::PerHead, CompactionClock::Synchronous] {
            let mut wrapped = compacting(2, configuration(clock, CompactionMode::Shadow, 2));
            let mut reference = plain(2);
            start_match_counting();
            train(&mut wrapped, 8);
            let counters = read_match_counters().unwrap();
            stop_match_counting();
            train_plain(&mut reference, 8);
            assert_eq!(wrapped.agent().snapshot(), reference.snapshot());
            let counts = wrapped.agent().counts();
            assert_eq!(
                counters.formations,
                counts.environment_steps
                    + 3 * counts.full
                    + 2 * counts.values_only
                    + wrapped.agent().priority_counts().value_queries
            );
            let reductions: usize = (0..2)
                .map(|head| wrapped.reduction_records(head).len())
                .sum();
            assert!(reductions > 0);
            for head in 0..2 {
                for record in wrapped.reduction_records(head) {
                    assert_eq!(record.mode, CompactionMode::Shadow);
                    assert!(!record.applied);
                    assert_eq!(record.size_after, record.size_before);
                }
            }
        }
    }

    #[test]
    fn apply_with_the_exact_guard_is_deterministic_and_neutral_for_the_acceptance_distribution() {
        let run = || {
            let mut wrapped = compacting(
                2,
                configuration(CompactionClock::PerHead, CompactionMode::Apply, 2),
            );
            train(&mut wrapped, 12);
            wrapped
        };
        let (first, second) = (run(), run());
        assert_eq!(first.snapshot(), second.snapshot());
        let removed: usize = (0..2)
            .flat_map(|head| first.reduction_records(head))
            .map(|record| record.removals.total())
            .sum();
        assert!(removed > 0);
        assert!(!first.acceptance_audits().is_empty());
        for audit in first.acceptance_audits() {
            assert_eq!(audit.before, audit.after);
            assert!(!audit.vetoed);
        }
        for head in 0..2 {
            for record in first.reduction_records(head) {
                assert!(record.accepted);
                assert_eq!(record.applied, record.removals.total() > 0);
                assert_eq!(
                    record.size_before - record.removals.total(),
                    record.size_after
                );
            }
        }
    }

    #[test]
    fn per_head_clocks_follow_each_heads_own_interactions_and_synchronous_fires_everywhere() {
        let mut per_head = compacting(
            2,
            configuration(CompactionClock::PerHead, CompactionMode::Apply, 6),
        );
        train(&mut per_head, 10);
        for head in 0..2 {
            let records = per_head.reduction_records(head);
            assert!(!records.is_empty());
            for record in records {
                assert_eq!(record.interaction % 3, 0);
                assert!(record.interaction >= 6);
            }
        }
        assert_eq!(per_head.head_controllers().unwrap().len(), 2);

        let mut synchronous = compacting(
            2,
            configuration(CompactionClock::Synchronous, CompactionMode::Apply, 6),
        );
        train(&mut synchronous, 10);
        assert!(synchronous.synchronous_policy().unwrap().fires() > 0);
        let lengths: Vec<_> = (0..2)
            .map(|head| synchronous.reduction_records(head).len())
            .collect();
        assert_eq!(lengths[0], lengths[1]);
        assert_eq!(lengths[0] as u64, synchronous.synchronous_policy().unwrap().fires());
        for (left, right) in synchronous
            .reduction_records(0)
            .iter()
            .zip(synchronous.reduction_records(1))
        {
            assert_eq!(left.interaction, right.interaction);
            assert_eq!(left.observation, right.observation);
        }
        assert!(synchronous.head_controllers().is_none());
    }

    #[test]
    fn a_single_intervention_fires_once_per_clock_unit() {
        let policy = fixed_strength_policy(Once::new(6), exact_strength()).unwrap();
        let single = |clock| {
            CompactingConfiguration::new(clock, core_parameters(CompactionMode::Apply, policy.clone()))
        };
        let mut per_head = compacting(2, single(CompactionClock::PerHead));
        train(&mut per_head, 14);
        let mut synchronous = compacting(2, single(CompactionClock::Synchronous));
        train(&mut synchronous, 14);
        for head in 0..2 {
            assert_eq!(per_head.reduction_records(head).len(), 1);
            assert_eq!(synchronous.reduction_records(head).len(), 1);
        }
        assert_eq!(synchronous.synchronous_policy().unwrap().fires(), 1);
    }

    #[test]
    fn matching_cost_per_head_adds_up_to_the_training_formations() {
        let mut wrapped = compacting(
            4,
            configuration(CompactionClock::PerHead, CompactionMode::Shadow, 3),
        );
        train(&mut wrapped, 12);
        let counts = wrapped.agent().counts();
        let formations: u64 = (0..4).map(|head| wrapped.head_cost(head).match_formations).sum();
        assert_eq!(
            formations,
            counts.environment_steps
                + 3 * counts.full
                + 2 * counts.values_only
                + wrapped.agent().priority_counts().value_queries
        );
        for head in 0..4 {
            let cost = wrapped.head_cost(head);
            assert!(cost.classifier_tests >= cost.match_formations);
            let own = wrapped.agent().heads()[head].counts();
            assert_eq!(own.environment_steps, 9);
        }
        let observed = wrapped.reduction_records(0).last().unwrap().observation;
        assert!(observed.match_formations_since_reduction > 0);
        assert!(observed.classifier_tests_since_reduction > 0);
    }

    #[test]
    fn evidence_is_the_stored_transitions_joined_with_their_original_goals() {
        let mut wrapped = compacting(2, CompactingConfiguration::off());
        train(&mut wrapped, 3);
        let store = wrapped.agent().store();
        let source = StoredEvidence { store };
        let everything: Vec<Transition<2>> =
            source.sample(1000, &mut ChaChaRandomSource::from_seed(1));
        assert_eq!(everything.len(), store.len());
        for transition in &everything {
            assert_eq!(transition.p0.symbols[1], Symbol::Token(9));
            assert_eq!(transition.p1.symbols[1], Symbol::Token(9));
        }
        let first = store.episodes().next().unwrap();
        assert_eq!(everything[0].p0.symbols[0], first.start().observation.symbols[0]);
        assert_eq!(everything[0].action, first.steps()[0].action);
        let few: Vec<Transition<2>> = source.sample(4, &mut ChaChaRandomSource::from_seed(1));
        assert_eq!(few.len(), 4);
    }

    #[test]
    fn the_uncounted_action_value_matches_the_priority_value_without_touching_the_counters() {
        let config = learning_configuration();
        let mut population = redundant_head(&config);
        let mut carrier = Classifier::general(Some(0), &config);
        carrier.condition.symbols = [Symbol::Wildcard, Symbol::Token(9)];
        carrier.effect.symbols[0] = Symbol::Token(1);
        carrier.q = 1.0;
        carrier.r = 100.0;
        population.insert(carrier);
        let sample = ReplaySample {
            state: perception([0, 9]),
            action: 0,
            reward: 0.0,
            next_state: perception([1, 9]),
            done: false,
        };
        start_match_counting();
        let uncounted = action_value_uncounted(&population, &sample);
        assert_eq!(read_match_counters().unwrap().formations, 0);
        let counted = action_value(&population, &sample);
        assert_eq!(read_match_counters().unwrap().formations, 1);
        stop_match_counting();
        assert_eq!(uncounted.to_bits(), counted.to_bits());
        assert_eq!(uncounted, 100.0);
    }

    #[test]
    fn disagreement_probes_follow_the_episode_clock_in_every_mode() {
        let probe = DisagreementProbe::new(2, 6);
        let mut off = compacting(2, CompactingConfiguration::off().with_disagreement_probe(probe));
        let mut reference = plain(2);
        train(&mut off, 6);
        train_plain(&mut reference, 6);
        assert_eq!(off.agent().snapshot(), reference.snapshot());
        let episodes: Vec<_> = off.disagreement_records().iter().map(|r| r.episodes).collect();
        assert_eq!(episodes, vec![2, 4, 6]);
        let last = off.disagreement_records().last().unwrap();
        assert_eq!(last.head_sizes.len(), 2);
        assert!(last.summary.mean_acceptance > 0.0 && last.summary.mean_acceptance <= 1.0);
        assert!(off.probe_work() > 0);
    }

    #[test]
    fn an_acceptance_veto_rolls_back_a_pass_that_moves_the_acceptance_distribution() {
        let thresholds = DynamicThresholds::new(0, 0, 2.0);
        let parameters = CompactionParameters::builder(0.3, periodic_policy(3))
            .operator(Operator::DynamicThresholds(thresholds))
            .evidence_sample(8)
            .build()
            .unwrap();
        let limits = AcceptanceLimits::new(0.0, 0.0);
        let mut wrapped = compacting(
            2,
            CompactingConfiguration::new(CompactionClock::PerHead, parameters)
                .with_acceptance_limits(limits),
        );
        train(&mut wrapped, 6);
        let vetoed: Vec<_> = wrapped
            .acceptance_audits()
            .iter()
            .filter(|audit| audit.vetoed)
            .collect();
        assert!(!vetoed.is_empty());
        for audit in vetoed {
            let record = &wrapped.reduction_records(audit.head)[audit.sequence as usize];
            assert!(record.check_vetoed && !record.applied && !record.accepted);
            assert_eq!(record.check_work, audit.work);
        }
    }

    #[test]
    fn cost_benefit_policies_run_on_the_derived_matching_cost() {
        let policy = fixed_strength_policy(CostBenefit::new(300, 50, 1).unwrap(), exact_strength())
            .unwrap();
        let mut wrapped = compacting(
            2,
            CompactingConfiguration::new(
                CompactionClock::PerHead,
                core_parameters(CompactionMode::Shadow, policy),
            ),
        );
        train(&mut wrapped, 12);
        assert!((0..2).any(|head| !wrapped.reduction_records(head).is_empty()));
    }
}
