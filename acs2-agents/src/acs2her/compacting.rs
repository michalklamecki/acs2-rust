use acs2_core::compaction::COMPACTION_STREAM;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_trajectory::SAMPLER_STREAM;

use super::{Agent, Preset};
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
        preset: Preset,
        compacting: CompactingConfiguration,
        seed: u64,
        agent_stream: u64,
    ) -> Self {
        Self::new(
            Agent::new(
                config,
                replay,
                preset,
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
            Preset::Acs2Her,
            ChaChaRandomSource::from_seed_and_stream(7, 1),
            ChaChaRandomSource::from_seed_and_stream(7, SAMPLER_STREAM),
        )
    }

    fn seeded(configuration: CompactingConfiguration) -> Compacting {
        Compacting::seeded(
            learning_configuration(false),
            episodic_configuration(),
            Preset::Acs2Her,
            configuration,
            7,
            1,
        )
    }

    #[test]
    fn the_seeded_variant_with_compaction_off_is_the_plain_hindsight_agent() {
        let mut wrapped = seeded(CompactingConfiguration::off());
        let mut reference = plain();
        let mut env = CycleEnvironment::truncating();
        let mut other = CycleEnvironment::truncating();
        for episode in 0..8 {
            wrapped.train_episode(&mut env, episode * 3);
            reference
                .learner_mut()
                .train_episode(&mut other, episode * 3);
        }
        assert_eq!(wrapped.agent().learner().snapshot(), reference.learner().snapshot());
        assert_eq!(wrapped.name(), "acs2her");
    }

    #[test]
    fn the_seeded_variant_applies_deterministically_on_the_compaction_stream() {
        let run = || {
            let configuration = CompactingConfiguration::new(subsuming_parameters(
                CompactionMode::Apply,
                3,
                true,
            ));
            let mut agent = seeded(configuration);
            let mut env = CycleEnvironment::truncating();
            for episode in 0..8 {
                agent.train_episode(&mut env, episode * 3);
            }
            agent
        };
        let (first, second) = (run(), run());
        assert_eq!(first.snapshot(), second.snapshot());
        assert!(!first.reduction_records().is_empty());
        assert_eq!(first.mode(), CompactionMode::Apply);
    }
}
