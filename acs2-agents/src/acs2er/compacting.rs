use core::ops::Range;

use acs2_core::checkpoint::Checkpointed;
use acs2_core::compaction::{
    subsample_indices, CompactionController, CompactionMode, CompactionParameters,
    EpisodeFormations, EvidenceSource, MatchingCost, ReductionRecord, SingleHeadCompaction,
    TaskGeometry, COMPACTION_STREAM,
};
use acs2_core::learning::action_selection::{ActionSelector, EpsilonGreedy};
use acs2_core::learning::rl::{BootstrapEstimator, MaxFitnessBootstrap};
use acs2_core::ports::environment::Environment;
use acs2_core::representation::population::Population;
use acs2_core::support::config::Configuration;
use acs2_core::support::knowledge::Transition;
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::trial::{LearningAgent, TrialMetrics, TruncationMode};

use super::replay::{ReplayConfiguration, ReplayMemory};
use super::Agent;

#[derive(Clone, Debug, PartialEq)]
pub struct CompactingConfiguration {
    compaction: CompactionParameters,
}

impl CompactingConfiguration {
    pub fn new(compaction: CompactionParameters) -> Self {
        Self { compaction }
    }

    pub fn off() -> Self {
        Self::new(CompactionParameters::off())
    }

    pub fn compaction(&self) -> &CompactionParameters {
        &self.compaction
    }
}

pub struct MemoryEvidence<'a, const N: usize> {
    memory: &'a ReplayMemory<N>,
}

impl<'a, const N: usize> MemoryEvidence<'a, N> {
    pub fn new(memory: &'a ReplayMemory<N>) -> Self {
        Self { memory }
    }
}

impl<const N: usize> EvidenceSource<N> for MemoryEvidence<'_, N> {
    fn sample(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<Transition<N>> {
        subsample_indices(self.memory.len(), size, rng)
            .into_iter()
            .map(|index| {
                let sample = self.memory.get(index);
                Transition::new(sample.state, sample.action, sample.next_state)
            })
            .collect()
    }
}

pub fn episode_formations(
    steps: u64,
    memory_before: usize,
    replay: &ReplayConfiguration,
    capacity: usize,
) -> EpisodeFormations {
    let mut length = memory_before;
    let mut learning = 0u64;
    for _ in 0..steps {
        length = (length + 1).min(capacity);
        if length >= replay.min_samples {
            learning += 3 * replay.samples_number.min(length) as u64;
        }
    }
    EpisodeFormations {
        environment: steps,
        learning,
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
        replay: ReplayConfiguration,
        truncation: TruncationMode,
        compacting: CompactingConfiguration,
        goal_attributes: Range<usize>,
        seed: u64,
        agent_stream: u64,
    ) -> Self {
        Self::with_goal_attributes(
            Agent::new(
                config,
                replay,
                ChaChaRandomSource::from_seed_and_stream(seed, agent_stream),
            )
            .with_truncation_mode(truncation),
            compacting,
            ChaChaRandomSource::from_seed_and_stream(seed, COMPACTION_STREAM),
            goal_attributes,
        )
    }
}

impl<const N: usize, R: RandomSource, C: RandomSource> CompactingAgent<N, R, C> {
    pub fn new(agent: Agent<N, R>, configuration: CompactingConfiguration, rng: C) -> Self {
        Self::with_goal_attributes(agent, configuration, rng, 0..0)
    }

    pub fn with_goal_attributes(
        agent: Agent<N, R>,
        configuration: CompactingConfiguration,
        rng: C,
        goal_attributes: Range<usize>,
    ) -> Self {
        let geometry = TaskGeometry {
            actions: agent.config().number_of_possible_actions,
            goal_attributes,
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
            "{:?}|{:?}|{:?}|{:?}",
            self.agent.capture(),
            self.agent.config(),
            self.agent.replay_config(),
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
        let memory_before = self.agent.replay_memory().len();
        let metrics = self.agent.run_explore_trial(env, selector, bootstrap, time);
        self.interactions += u64::from(metrics.steps);
        let formations = episode_formations(
            u64::from(metrics.steps),
            memory_before,
            self.agent.replay_config(),
            self.agent.replay_memory().max_size(),
        );
        self.compaction
            .charge_episode(formations, started, self.agent.population().len());
        let replacement = self.compaction.reduce(
            self.agent.population(),
            self.interactions,
            &MemoryEvidence::new(self.agent.replay_memory()),
        );
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
    use acs2_core::support::measurement::{
        read_match_counters, start_match_counting, stop_match_counting,
    };

    use crate::episodic_replay::test_environment::{
        learning_configuration, redundant_population, subsuming_parameters, CycleEnvironment,
    };

    type Plain = Agent<2, ChaChaRandomSource>;
    type Compacting = CompactingAgent<2, ChaChaRandomSource, ChaChaRandomSource>;

    fn replay() -> ReplayConfiguration {
        ReplayConfiguration {
            buffer_size: 5,
            min_samples: 2,
            samples_number: 3,
        }
    }

    fn plain(do_ga: bool, truncation: TruncationMode) -> Plain {
        let config = learning_configuration(do_ga);
        let population = redundant_population(&config);
        Plain::with_population(
            config,
            replay(),
            ChaChaRandomSource::from_seed_and_stream(42, 1),
            population,
        )
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

    fn configuration(mode: CompactionMode, enforced_audit: bool) -> CompactingConfiguration {
        CompactingConfiguration::new(subsuming_parameters(mode, 3, enforced_audit))
    }

    fn train(agent: &mut Compacting, episodes: u64) {
        let mut env = CycleEnvironment::truncating();
        for episode in 0..episodes {
            agent.train_episode(&mut env, episode * 3);
        }
    }

    fn train_plain(agent: &mut Plain, episodes: u64) {
        let selector = EpsilonGreedy {
            number_of_possible_actions: 2,
            epsilon: 0.5,
        };
        let mut env = CycleEnvironment::truncating();
        for episode in 0..episodes {
            agent.run_explore_trial(&mut env, &selector, &MaxFitnessBootstrap, episode * 3);
        }
    }

    fn state(agent: &Plain) -> String {
        format!("{:?}", agent.capture())
    }

    #[test]
    fn off_gives_exactly_the_learning_state_of_the_plain_agent() {
        let mut wrapped = compacting(false, TruncationMode::Bootstrap, CompactingConfiguration::off());
        let mut reference = plain(false, TruncationMode::Bootstrap);
        train(&mut wrapped, 9);
        train_plain(&mut reference, 9);
        assert_eq!(state(wrapped.agent()), state(&reference));
        assert!(wrapped.reduction_records().is_empty());
        assert_eq!(wrapped.cost(), MatchingCost::default());
        assert_eq!(wrapped.compaction_random_source().capture_state().unwrap().word_pos, 0);
    }

    #[test]
    fn shadow_follows_the_plain_learning_trajectory_and_audits_on_the_replay_memory() {
        let mut wrapped = compacting(false, TruncationMode::Bootstrap, configuration(CompactionMode::Shadow, true));
        let mut reference = plain(false, TruncationMode::Bootstrap);
        train(&mut wrapped, 8);
        train_plain(&mut reference, 8);
        assert_eq!(state(wrapped.agent()), state(&reference));
        assert!(!wrapped.reduction_records().is_empty());
        for record in wrapped.reduction_records() {
            assert!(!record.applied);
            assert_eq!(record.size_after, record.size_before);
            assert!(record.evidence_size > 0 && record.evidence_size <= 5);
        }
    }

    #[test]
    fn apply_with_the_exact_guard_is_deterministic_and_populates_the_records() {
        let run = |audited| {
            let mut wrapped = compacting(
                false,
                TruncationMode::Bootstrap,
                configuration(CompactionMode::Apply, audited),
            );
            train(&mut wrapped, 12);
            wrapped
        };
        for audited in [false, true] {
            let (first, second) = (run(audited), run(audited));
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
                assert_eq!(record.size_before - record.removals.total(), record.size_after);
                assert_eq!(record.audit.is_some(), audited && record.removals.total() > 0);
            }
        }
    }

    #[test]
    fn derived_formations_equal_the_recorded_ones_across_warmup_and_a_full_buffer() {
        for mode in [CompactionMode::Shadow, CompactionMode::Apply] {
            for do_ga in [false, true] {
                let mut wrapped = compacting(do_ga, TruncationMode::Pyalcs, configuration(mode, true));
                start_match_counting();
                train(&mut wrapped, 7);
                let counters = read_match_counters().unwrap();
                stop_match_counting();
                assert_eq!(wrapped.cost().match_formations, counters.formations);
                assert!(wrapped.cost().classifier_tests >= wrapped.cost().match_formations);
            }
        }
    }

    #[test]
    fn episode_formations_follow_the_warmup_the_buffer_and_the_draw_count() {
        let formations = episode_formations(3, 0, &replay(), 5);
        assert_eq!(formations.environment, 3);
        assert_eq!(formations.learning, 3 * (2 + 3));
        let full = episode_formations(3, 5, &replay(), 5);
        assert_eq!(full.learning, 3 * 3 * 3);
    }

    #[test]
    fn the_evidence_is_the_replay_memory_transitions() {
        let mut wrapped = compacting(false, TruncationMode::Bootstrap, CompactingConfiguration::off());
        train(&mut wrapped, 3);
        let memory = wrapped.agent().replay_memory();
        let source = MemoryEvidence::new(memory);
        let everything: Vec<Transition<2>> =
            source.sample(1000, &mut ChaChaRandomSource::from_seed(1));
        assert_eq!(everything.len(), memory.len());
        let first = memory.get(0);
        assert_eq!(everything[0].p0, first.state);
        assert_eq!(everything[0].action, first.action);
        assert_eq!(everything[0].p1, first.next_state);
        let few: Vec<Transition<2>> = source.sample(2, &mut ChaChaRandomSource::from_seed(1));
        assert_eq!(few.len(), 2);
    }
}
