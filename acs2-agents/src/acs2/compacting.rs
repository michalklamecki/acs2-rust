use core::fmt;
use std::error::Error;

use acs2_core::checkpoint::Checkpointed;
use acs2_core::compaction::{
    CompactionController, CompactionMode, CompactionParameters, EpisodeFormations, MatchingCost,
    NoEvidence, OperatorKind, ReductionRecord, SingleHeadCompaction, TaskGeometry,
    COMPACTION_STREAM,
};
use acs2_core::learning::action_selection::{ActionSelector, EpsilonGreedy};
use acs2_core::learning::rl::{BootstrapEstimator, MaxFitnessBootstrap};
use acs2_core::ports::environment::{Environment, StepOutcome};
use acs2_core::representation::perception::Perception;
use acs2_core::representation::population::Population;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::trial::{LearningAgent, TrialMetrics, TruncationMode};

use super::Agent;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompactingRefusal {
    EvidenceOperator(OperatorKind),
    EvidenceAudit,
}

impl fmt::Display for CompactingRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvidenceOperator(kind) => write!(
                formatter,
                "ACS2 keeps no experience memory, so operator {kind:?} has no evidence to work on"
            ),
            Self::EvidenceAudit => write!(
                formatter,
                "ACS2 keeps no experience memory, so a before and after audit has no evidence to run on"
            ),
        }
    }
}

impl Error for CompactingRefusal {}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactingConfiguration {
    compaction: CompactionParameters,
}

impl CompactingConfiguration {
    pub fn new(compaction: CompactionParameters) -> Result<Self, CompactingRefusal> {
        if let Some(kind) = compaction.evidence_operator() {
            return Err(CompactingRefusal::EvidenceOperator(kind));
        }
        if compaction.audit().thresholds().is_some() {
            return Err(CompactingRefusal::EvidenceAudit);
        }
        Ok(Self { compaction })
    }

    pub fn off() -> Self {
        Self {
            compaction: CompactionParameters::off(),
        }
    }

    pub fn compaction(&self) -> &CompactionParameters {
        &self.compaction
    }
}

pub fn episode_formations(steps: u64, do_ga: bool, terminal: bool) -> EpisodeFormations {
    let revisits = steps.saturating_sub(1) * (1 + u64::from(do_ga));
    let closing = if terminal { 0 } else { 2 };
    EpisodeFormations {
        environment: steps,
        learning: revisits + closing,
    }
}

struct EndObserver<'a, E> {
    inner: &'a mut E,
    terminated: bool,
    truncated: bool,
}

impl<E: Environment<N>, const N: usize> Environment<N> for EndObserver<'_, E> {
    fn reset(&mut self) -> Perception<N> {
        self.inner.reset()
    }

    fn step(&mut self, action: usize) -> StepOutcome<N> {
        let outcome = self.inner.step(action);
        self.terminated = outcome.terminated;
        self.truncated = outcome.truncated;
        outcome
    }
}

pub struct CompactingAgent<const N: usize, R: RandomSource, C> {
    agent: Agent<N, R>,
    configuration: CompactingConfiguration,
    compaction: SingleHeadCompaction<N, C>,
    interactions: u64,
}

impl<const N: usize> CompactingAgent<N, ChaChaRandomSource, ChaChaRandomSource> {
    pub fn seeded(
        config: Configuration,
        truncation: TruncationMode,
        compacting: CompactingConfiguration,
        seed: u64,
        agent_stream: u64,
    ) -> Self {
        Self::new(
            Agent::new(
                config,
                ChaChaRandomSource::from_seed_and_stream(seed, agent_stream),
            )
            .with_truncation_mode(truncation),
            compacting,
            ChaChaRandomSource::from_seed_and_stream(seed, COMPACTION_STREAM),
        )
    }
}

impl<const N: usize, R: RandomSource, C: RandomSource> CompactingAgent<N, R, C> {
    pub fn new(agent: Agent<N, R>, configuration: CompactingConfiguration, rng: C) -> Self {
        let geometry = TaskGeometry {
            actions: agent.config().number_of_possible_actions,
            goal_attributes: 0..0,
        };
        let compaction =
            SingleHeadCompaction::new(configuration.compaction.clone(), geometry, rng);
        Self {
            agent,
            configuration,
            compaction,
            interactions: 0,
        }
    }

    pub fn agent(&self) -> &Agent<N, R> {
        &self.agent
    }

    pub fn configuration(&self) -> &CompactingConfiguration {
        &self.configuration
    }

    pub fn mode(&self) -> CompactionMode {
        self.compaction.mode()
    }

    pub fn interactions(&self) -> u64 {
        self.interactions
    }

    pub fn cost(&self) -> MatchingCost {
        self.compaction.cost()
    }

    pub fn controller(&self) -> &CompactionController<N> {
        self.compaction.controller()
    }

    pub fn reduction_records(&self) -> &[ReductionRecord] {
        self.compaction.records()
    }

    pub fn compaction_random_source(&self) -> &C {
        self.compaction.random_source()
    }

    pub fn learning_snapshot(&self) -> String {
        format!(
            "{:?}|{:?}|{:?}",
            self.agent.capture(),
            self.agent.config(),
            self.agent.truncation_mode()
        )
    }

    pub fn snapshot(&self) -> String {
        format!(
            "{}|{:?}|{}|{}",
            self.learning_snapshot(),
            self.configuration,
            self.compaction.snapshot(),
            self.interactions
        )
    }

    pub fn train_episode<E: Environment<N>>(&mut self, env: &mut E, time: u64) -> TrialMetrics {
        let selector = EpsilonGreedy {
            number_of_possible_actions: self.agent.config().number_of_possible_actions,
            epsilon: self.agent.config().epsilon,
        };
        self.run_explore_trial(env, &selector, &MaxFitnessBootstrap, time)
    }

    fn explore<E, S, B>(
        &mut self,
        env: &mut E,
        selector: &S,
        bootstrap: &B,
        time: u64,
    ) -> TrialMetrics
    where
        E: Environment<N>,
        S: ActionSelector<N>,
        B: BootstrapEstimator<N>,
    {
        if !self.compaction.is_active() {
            let metrics = self.agent.run_explore_trial(env, selector, bootstrap, time);
            self.interactions += u64::from(metrics.steps);
            return metrics;
        }
        let started = self.agent.population().len();
        let mut observed = EndObserver {
            inner: env,
            terminated: false,
            truncated: false,
        };
        let metrics = self
            .agent
            .run_explore_trial(&mut observed, selector, bootstrap, time);
        let terminal = self
            .agent
            .truncation_mode()
            .is_terminal(observed.terminated, observed.truncated);
        self.interactions += u64::from(metrics.steps);
        self.compaction.charge_episode(
            episode_formations(u64::from(metrics.steps), self.agent.config().do_ga, terminal),
            started,
            self.agent.population().len(),
        );
        let replacement =
            self.compaction
                .reduce(self.agent.population(), self.interactions, &NoEvidence);
        if let Some(population) = replacement {
            self.agent.replace_population(population);
        }
        metrics
    }
}

impl<const N: usize, R: RandomSource, C: RandomSource> LearningAgent<N>
    for CompactingAgent<N, R, C>
{
    fn population(&self) -> &Population<N> {
        self.agent.population()
    }

    fn config(&self) -> &Configuration {
        self.agent.config()
    }

    fn run_explore_trial<E, S, B>(
        &mut self,
        env: &mut E,
        selector: &S,
        bootstrap: &B,
        time: u64,
    ) -> TrialMetrics
    where
        E: Environment<N>,
        S: ActionSelector<N>,
        B: BootstrapEstimator<N>,
    {
        self.explore(env, selector, bootstrap, time)
    }

    fn run_exploit_trial<E, B>(&mut self, env: &mut E, bootstrap: &B, time: u64) -> TrialMetrics
    where
        E: Environment<N>,
        B: BootstrapEstimator<N>,
    {
        self.agent.run_exploit_trial(env, bootstrap, time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acs2_core::compaction::trigger::Periodic;
    use acs2_core::compaction::{
        fixed_strength_policy, AuditPolicy, AuditThresholds, CapacityConfiguration,
        CompactionStrength, DynamicThresholds, Operator,
    };
    use acs2_core::support::measurement::{
        read_match_counters, start_match_counting, stop_match_counting,
    };

    use crate::episodic_replay::test_environment::{
        learning_configuration, redundant_population, subsuming_parameters, CycleEnvironment,
    };

    type Plain = Agent<2, ChaChaRandomSource>;
    type Compacting = CompactingAgent<2, ChaChaRandomSource, ChaChaRandomSource>;

    fn plain(do_ga: bool, truncation: TruncationMode) -> Plain {
        let config = learning_configuration(do_ga);
        let population = redundant_population(&config);
        Plain::with_population(config, ChaChaRandomSource::from_seed_and_stream(42, 1), population)
            .with_truncation_mode(truncation)
    }

    fn compacting(
        do_ga: bool,
        truncation: TruncationMode,
        configuration: CompactingConfiguration,
    ) -> Compacting {
        Compacting::new(
            plain(do_ga, truncation),
            configuration,
            ChaChaRandomSource::from_seed_and_stream(42, COMPACTION_STREAM),
        )
    }

    fn subsuming(mode: CompactionMode, every: u64) -> CompactingConfiguration {
        CompactingConfiguration::new(subsuming_parameters(mode, every, false)).unwrap()
    }

    fn train(agent: &mut Compacting, episodes: u64, terminating: bool) {
        let mut env = environment(terminating);
        for episode in 0..episodes {
            agent.train_episode(&mut env, episode * 3);
        }
    }

    fn train_plain(agent: &mut Plain, episodes: u64, terminating: bool) {
        let selector = EpsilonGreedy {
            number_of_possible_actions: 2,
            epsilon: 0.5,
        };
        let mut env = environment(terminating);
        for episode in 0..episodes {
            agent.run_explore_trial(&mut env, &selector, &MaxFitnessBootstrap, episode * 3);
        }
    }

    fn environment(terminating: bool) -> CycleEnvironment {
        if terminating {
            CycleEnvironment::terminating()
        } else {
            CycleEnvironment::truncating()
        }
    }

    fn state(agent: &Plain) -> String {
        format!("{:?}", agent.capture())
    }

    #[test]
    fn operators_and_audits_that_need_evidence_are_refused() {
        let strength = CompactionStrength {
            value_tolerance: 0.0,
            maturity_experience: 20,
            idle_audits: 1,
        };
        let policy = fixed_strength_policy(Periodic::new(3).unwrap(), strength).unwrap();
        let builder = || CompactionParameters::builder(0.3, policy.clone());
        let idle = builder()
            .operator(Operator::IdlePruning { min_experience: 5 })
            .evidence_sample(4)
            .build()
            .unwrap();
        assert_eq!(
            CompactingConfiguration::new(idle),
            Err(CompactingRefusal::EvidenceOperator(OperatorKind::IdlePruning))
        );
        let capacity = builder()
            .operator(Operator::Capacity(CapacityConfiguration::new(3)))
            .evidence_sample(4)
            .build()
            .unwrap();
        assert_eq!(
            CompactingConfiguration::new(capacity),
            Err(CompactingRefusal::EvidenceOperator(OperatorKind::Capacity))
        );
        for audit in [
            AuditPolicy::Observe(AuditThresholds::lossless()),
            AuditPolicy::Enforce(AuditThresholds::lossless()),
        ] {
            let audited = builder()
                .operator(Operator::MatureSubsumption { guarded: true })
                .audit(audit)
                .evidence_sample(4)
                .build()
                .unwrap();
            assert_eq!(
                CompactingConfiguration::new(audited),
                Err(CompactingRefusal::EvidenceAudit)
            );
        }
        let allowed = [
            Operator::MatureSubsumption { guarded: true },
            Operator::CracsSubsumption { guarded: true },
            Operator::DynamicThresholds(DynamicThresholds::new(0, 0, 2.0)),
        ];
        for operator in allowed {
            let parameters = builder().operator(operator).build().unwrap();
            assert!(CompactingConfiguration::new(parameters).is_ok());
        }
        assert!(CompactingConfiguration::new(CompactionParameters::off()).is_ok());
        assert!(!CompactingRefusal::EvidenceAudit.to_string().is_empty());
    }

    #[test]
    fn off_gives_exactly_the_learning_state_of_the_plain_agent() {
        for (do_ga, truncation) in [
            (false, TruncationMode::Bootstrap),
            (true, TruncationMode::Pyalcs),
        ] {
            let mut wrapped = compacting(do_ga, truncation, CompactingConfiguration::off());
            let mut reference = plain(do_ga, truncation);
            train(&mut wrapped, 9, false);
            train_plain(&mut reference, 9, false);
            assert_eq!(state(wrapped.agent()), state(&reference));
            assert!(wrapped.reduction_records().is_empty());
            assert_eq!(wrapped.cost(), MatchingCost::default());
            assert_eq!(wrapped.compaction_random_source().capture_state().unwrap().word_pos, 0);
            assert_eq!(wrapped.interactions(), 27);
        }
    }

    #[test]
    fn shadow_follows_the_plain_learning_trajectory_and_records_without_applying() {
        let mut wrapped = compacting(false, TruncationMode::Bootstrap, subsuming(CompactionMode::Shadow, 3));
        let mut reference = plain(false, TruncationMode::Bootstrap);
        train(&mut wrapped, 8, false);
        train_plain(&mut reference, 8, false);
        assert_eq!(state(wrapped.agent()), state(&reference));
        assert!(!wrapped.reduction_records().is_empty());
        for record in wrapped.reduction_records() {
            assert_eq!(record.mode, CompactionMode::Shadow);
            assert!(!record.applied);
            assert_eq!(record.size_after, record.size_before);
            assert_eq!(record.interaction % 3, 0);
        }
    }

    #[test]
    fn apply_with_the_exact_guard_is_deterministic_and_populates_the_records() {
        let run = || {
            let mut wrapped =
                compacting(false, TruncationMode::Bootstrap, subsuming(CompactionMode::Apply, 3));
            train(&mut wrapped, 12, false);
            wrapped
        };
        let (first, second) = (run(), run());
        assert_eq!(first.snapshot(), second.snapshot());
        assert!(!first.reduction_records().is_empty());
        let removed: usize = first
            .reduction_records()
            .iter()
            .map(|record| record.removals.total())
            .sum();
        assert!(removed > 0);
        for record in first.reduction_records() {
            assert!(record.accepted);
            assert_eq!(record.applied, record.removals.total() > 0);
            assert_eq!(record.size_before - record.removals.total(), record.size_after);
        }
        assert_eq!(first.controller().records().len(), first.reduction_records().len());
    }

    #[test]
    fn derived_formations_equal_the_recorded_ones_in_every_configuration() {
        for mode in [CompactionMode::Shadow, CompactionMode::Apply] {
            for do_ga in [false, true] {
                for truncation in [TruncationMode::Bootstrap, TruncationMode::Pyalcs] {
                    for terminating in [false, true] {
                        let mut wrapped = compacting(do_ga, truncation, subsuming(mode, 3));
                        start_match_counting();
                        train(&mut wrapped, 7, terminating);
                        let counters = read_match_counters().unwrap();
                        stop_match_counting();
                        assert_eq!(
                            wrapped.cost().match_formations,
                            counters.formations,
                            "{mode:?} ga {do_ga} {truncation:?} terminating {terminating}"
                        );
                        assert!(wrapped.cost().classifier_tests > 0);
                        assert!(wrapped.cost().classifier_tests >= wrapped.cost().match_formations);
                    }
                }
            }
        }
    }

    #[test]
    fn the_exploit_trial_is_untouched_and_not_a_compaction_boundary() {
        let mut wrapped = compacting(false, TruncationMode::Bootstrap, subsuming(CompactionMode::Apply, 3));
        let mut reference = plain(false, TruncationMode::Bootstrap);
        let mut env = CycleEnvironment::truncating();
        let mut other = CycleEnvironment::truncating();
        wrapped.run_exploit_trial(&mut env, &MaxFitnessBootstrap, 0);
        reference.run_exploit_trial(&mut other, &MaxFitnessBootstrap, 0);
        assert_eq!(state(wrapped.agent()), state(&reference));
        assert!(wrapped.reduction_records().is_empty());
        assert_eq!(wrapped.interactions(), 0);
    }
}
