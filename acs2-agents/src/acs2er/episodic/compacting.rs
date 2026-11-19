use acs2_core::compaction::COMPACTION_STREAM;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_trajectory::SAMPLER_STREAM;

use super::Agent;
use crate::episodic_replay::compacting::CompactingEpisodic;
use crate::episodic_replay::EpisodicConfiguration;

pub use crate::episodic_replay::compacting::CompactingConfiguration;

pub type CompactingAgent<const S: usize, const G: usize, const M: usize, R, P, C> =
    CompactingEpisodic<S, G, M, Agent<S, G, M, R, P>, C>;

impl<const S: usize, const G: usize, const M: usize>
    CompactingAgent<
        S,
        G,
        M,
        ChaChaRandomSource,
        ChaChaRandomSource,
        ChaChaRandomSource,
    >
{
    pub fn seeded(
        config: Configuration,
        replay: EpisodicConfiguration,
        compacting: CompactingConfiguration,
        seed: u64,
        agent_stream: u64,
    ) -> Self {
        Self::new(
            Agent::new(
                config,
                replay,
                ChaChaRandomSource::from_seed_and_stream(seed, agent_stream),
                ChaChaRandomSource::from_seed_and_stream(seed, SAMPLER_STREAM),
            ),
            compacting,
            ChaChaRandomSource::from_seed_and_stream(seed, COMPACTION_STREAM),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acs2_core::compaction::CompactionMode;

    use crate::episodic_replay::test_environment::{
        episodic_configuration, learning_configuration, subsuming_parameters, CycleEnvironment,
    };
    use crate::episodic_replay::EpisodicAgent;

    type Plain = Agent<1, 1, 2, ChaChaRandomSource, ChaChaRandomSource>;
    type Compacting = CompactingAgent<
        1,
        1,
        2,
        ChaChaRandomSource,
        ChaChaRandomSource,
        ChaChaRandomSource,
    >;

    fn plain() -> Plain {
        Plain::new(
            learning_configuration(false),
            episodic_configuration(),
            ChaChaRandomSource::from_seed_and_stream(7, 1),
            ChaChaRandomSource::from_seed_and_stream(7, SAMPLER_STREAM),
        )
    }

    fn seeded(configuration: CompactingConfiguration) -> Compacting {
        Compacting::seeded(
            learning_configuration(false),
            episodic_configuration(),
            configuration,
            7,
            1,
        )
    }

    fn train(agent: &mut Compacting, episodes: u64) {
        let mut env = CycleEnvironment::truncating();
        for episode in 0..episodes {
            agent.train_episode(&mut env, episode * 3);
        }
    }

    #[test]
    fn off_and_shadow_follow_the_plain_episodic_agent() {
        for mode in [None, Some(CompactionMode::Shadow)] {
            let configuration = match mode {
                None => CompactingConfiguration::off(),
                Some(mode) => CompactingConfiguration::new(subsuming_parameters(mode, 3, true)),
            };
            let mut wrapped = seeded(configuration);
            let mut reference = plain();
            let mut other = CycleEnvironment::truncating();
            train(&mut wrapped, 8);
            for episode in 0..8 {
                reference
                    .learner_mut()
                    .train_episode(&mut other, episode * 3);
            }
            assert_eq!(wrapped.agent().learner().snapshot(), reference.learner().snapshot());
            assert_eq!(wrapped.reduction_records().is_empty(), mode.is_none());
            assert_eq!(wrapped.name(), "acs2er_episodic");
        }
    }

    #[test]
    fn apply_is_deterministic_and_populates_the_records() {
        let run = || {
            let configuration = CompactingConfiguration::new(subsuming_parameters(
                CompactionMode::Apply,
                3,
                true,
            ));
            let mut agent = seeded(configuration);
            train(&mut agent, 8);
            agent
        };
        let (first, second) = (run(), run());
        assert_eq!(first.snapshot(), second.snapshot());
        assert!(!first.reduction_records().is_empty());
    }
}
