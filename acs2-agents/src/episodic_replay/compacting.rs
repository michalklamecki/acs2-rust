use acs2_core::compaction::{
    CompactionController, CompactionMode, CompactionParameters, MatchingCost, ReductionRecord,
    SingleHeadCompaction, TaskGeometry,
};
use acs2_core::ports::goal::GoalLayout;
use acs2_core::representation::population::Population;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;

use super::evidence::StoredEvidence;
use super::{EpisodicAgent, ReplayEnvironment};

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

pub struct CompactingEpisodic<const S: usize, const G: usize, const M: usize, A, C> {
    agent: A,
    configuration: CompactingConfiguration,
    compaction: SingleHeadCompaction<M, C>,
}

impl<const S: usize, const G: usize, const M: usize, A, C> CompactingEpisodic<S, G, M, A, C>
where
    A: EpisodicAgent<S, G, M>,
    C: RandomSource,
{
    pub fn new(agent: A, configuration: CompactingConfiguration, rng: C) -> Self {
        let geometry = TaskGeometry {
            actions: agent.learner().config().number_of_possible_actions,
            goal_attributes: GoalLayout::<S, G, M>::goal_positions(),
        };
        let compaction =
            SingleHeadCompaction::new(configuration.compaction.clone(), geometry, rng);
        Self {
            agent,
            configuration,
            compaction,
        }
    }

    pub fn agent(&self) -> &A {
        &self.agent
    }

    pub fn configuration(&self) -> &CompactingConfiguration {
        &self.configuration
    }

    pub fn name(&self) -> &'static str {
        self.agent.name()
    }

    pub fn mode(&self) -> CompactionMode {
        self.compaction.mode()
    }

    pub fn population(&self) -> &Population<M> {
        self.agent.learner().population()
    }

    pub fn config(&self) -> &Configuration {
        self.agent.learner().config()
    }

    pub fn cost(&self) -> MatchingCost {
        self.compaction.cost()
    }

    pub fn controller(&self) -> &CompactionController<M> {
        self.compaction.controller()
    }

    pub fn reduction_records(&self) -> &[ReductionRecord] {
        self.compaction.records()
    }

    pub fn compaction_random_source(&self) -> &C {
        self.compaction.random_source()
    }

    pub fn snapshot(&self) -> String {
        format!(
            "{}|{:?}|{}",
            self.agent.learner().snapshot(),
            self.configuration,
            self.compaction.snapshot()
        )
    }

    pub fn train_episode(&mut self, env: &mut impl ReplayEnvironment<S, G, M>, time: u64) {
        if !self.compaction.is_active() {
            self.agent.learner_mut().train_episode(env, time);
            return;
        }
        let learner = self.agent.learner();
        let counts = learner.counts();
        let started = learner.population().len();
        self.agent.learner_mut().train_episode(env, time);
        let learner = self.agent.learner();
        self.compaction.charge_episode(
            learner.counts().formations_since(&counts),
            started,
            learner.population().len(),
        );
        let replacement = self.compaction.reduce(
            learner.population(),
            learner.counts().environment_steps,
            &StoredEvidence::new(learner.store()),
        );
        if let Some(population) = replacement {
            self.agent.learner_mut().replace_population(population);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acs2_core::compaction::CompactionStrength;
    use acs2_core::compaction::trigger::Periodic;
    use acs2_core::compaction::{
        fixed_strength_policy, AuditPolicy, AuditThresholds, EvidenceSource, Operator,
    };
    use acs2_core::representation::symbol::Symbol;
    use acs2_core::support::knowledge::Transition;
    use acs2_core::support::measurement::{
        read_match_counters, start_match_counting, stop_match_counting,
    };
    use acs2_core::support::rng::ChaChaRandomSource;
    use acs2_trajectory::SAMPLER_STREAM;

    use crate::acs2her::{Agent as HerAgent, Preset};
    use crate::episodic_replay::test_environment::{
        episodic_configuration, learning_configuration, redundant_population,
        subsuming_parameters, CycleEnvironment,
    };

    type Her = HerAgent<1, 1, 2, ChaChaRandomSource, ChaChaRandomSource>;
    type Compacting = CompactingEpisodic<1, 1, 2, Her, ChaChaRandomSource>;

    fn plain(preset: Preset, do_ga: bool) -> Her {
        let config = learning_configuration(do_ga);
        let mut agent = Her::new(
            config.clone(),
            episodic_configuration(),
            preset,
            ChaChaRandomSource::from_seed_and_stream(42, 1),
            ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
        );
        agent
            .learner_mut()
            .replace_population(redundant_population(&config));
        agent
    }

    fn compacting(preset: Preset, do_ga: bool, configuration: CompactingConfiguration) -> Compacting {
        Compacting::new(
            plain(preset, do_ga),
            configuration,
            ChaChaRandomSource::from_seed_and_stream(42, acs2_core::compaction::COMPACTION_STREAM),
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

    fn train_plain(agent: &mut Her, episodes: u64) {
        let mut env = CycleEnvironment::truncating();
        for episode in 0..episodes {
            agent.learner_mut().train_episode(&mut env, episode * 3);
        }
    }

    const PRESETS: [Preset; 2] = [Preset::Acs2Her, Preset::Acs2HerValues { proportion: 0.5 }];

    #[test]
    fn off_gives_the_snapshot_of_the_plain_agent() {
        for preset in PRESETS {
            let mut wrapped = compacting(preset, false, CompactingConfiguration::off());
            let mut reference = plain(preset, false);
            train(&mut wrapped, 9);
            train_plain(&mut reference, 9);
            assert_eq!(wrapped.agent().learner().snapshot(), reference.learner().snapshot());
            assert!(wrapped.reduction_records().is_empty());
            assert_eq!(wrapped.cost(), MatchingCost::default());
            assert_eq!(wrapped.compaction_random_source().capture_state().unwrap().word_pos, 0);
        }
    }

    #[test]
    fn shadow_follows_the_plain_learning_trajectory() {
        for preset in PRESETS {
            let mut wrapped = compacting(preset, false, configuration(CompactionMode::Shadow, true));
            let mut reference = plain(preset, false);
            train(&mut wrapped, 8);
            train_plain(&mut reference, 8);
            assert_eq!(wrapped.agent().learner().snapshot(), reference.learner().snapshot());
            assert!(!wrapped.reduction_records().is_empty());
            for record in wrapped.reduction_records() {
                assert!(!record.applied);
                assert_eq!(record.size_after, record.size_before);
                assert!(record.evidence_size > 0);
            }
        }
    }

    #[test]
    fn apply_with_the_exact_guard_is_deterministic_and_populates_the_records() {
        for preset in PRESETS {
            let run = |audited| {
                let mut wrapped =
                    compacting(preset, false, configuration(CompactionMode::Apply, audited));
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
                    assert_eq!(record.interaction % 3, 0);
                }
            }
        }
    }

    #[test]
    fn derived_formations_equal_the_recorded_ones_and_compaction_does_not_record() {
        for preset in PRESETS {
            for mode in [CompactionMode::Shadow, CompactionMode::Apply] {
                let mut wrapped = compacting(preset, true, configuration(mode, true));
                start_match_counting();
                train(&mut wrapped, 7);
                let counters = read_match_counters().unwrap();
                stop_match_counting();
                assert_eq!(wrapped.cost().match_formations, counters.formations);
                assert!(wrapped.cost().classifier_tests >= wrapped.cost().match_formations);
                let counts = wrapped.agent().learner().counts();
                assert_eq!(
                    counters.formations,
                    counts.environment_steps + 3 * counts.full + 2 * counts.values_only
                );
                assert!(counts.full > 0);
                assert_eq!(
                    counts.values_only > 0,
                    matches!(preset, Preset::Acs2HerValues { .. })
                );
            }
        }
    }

    #[test]
    fn the_trigger_reads_the_derived_cost() {
        let strength = CompactionStrength {
            value_tolerance: 0.0,
            maturity_experience: 20,
            idle_audits: 1,
        };
        let policy = fixed_strength_policy(Periodic::new(3).unwrap(), strength).unwrap();
        let parameters = acs2_core::compaction::CompactionParameters::builder(0.3, policy)
            .mode(CompactionMode::Shadow)
            .operator(Operator::IdlePruning { min_experience: 5 })
            .audit(AuditPolicy::Observe(AuditThresholds::lossless()))
            .evidence_sample(6)
            .build()
            .unwrap();
        let mut wrapped = compacting(
            Preset::Acs2Her,
            false,
            CompactingConfiguration::new(parameters),
        );
        train(&mut wrapped, 8);
        let observed = wrapped.reduction_records().last().unwrap().observation;
        assert!(observed.match_formations_since_reduction > 0);
        assert!(observed.classifier_tests_since_reduction >= observed.match_formations_since_reduction);
    }

    #[test]
    fn evidence_is_the_stored_transitions_joined_with_their_original_goals() {
        let mut wrapped = compacting(Preset::Acs2Her, false, CompactingConfiguration::off());
        train(&mut wrapped, 3);
        let store = wrapped.agent().learner().store();
        let source = StoredEvidence::new(store);
        let everything: Vec<Transition<2>> =
            source.sample(1000, &mut ChaChaRandomSource::from_seed(1));
        assert_eq!(everything.len(), store.len());
        for transition in &everything {
            assert_eq!(transition.p0.symbols[1], Symbol::Token(9));
            assert_eq!(transition.p1.symbols[1], Symbol::Token(9));
        }
    }
}
