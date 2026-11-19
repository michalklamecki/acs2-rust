#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MatchingCost {
    pub match_formations: u64,
    pub classifier_tests: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct EpisodeFormations {
    pub environment: u64,
    pub learning: u64,
}

impl EpisodeFormations {
    pub fn total(&self) -> u64 {
        self.environment + self.learning
    }
}

impl MatchingCost {
    pub fn charge_episode(
        &mut self,
        formations: EpisodeFormations,
        size_at_start: usize,
        size_at_end: usize,
    ) {
        let started = size_at_start as u64;
        let ended = size_at_end as u64;
        self.match_formations += formations.total();
        self.classifier_tests +=
            formations.environment * started + formations.learning * ((started + ended) / 2);
    }

    pub fn charge_queries(&mut self, queries: u64, population_size: usize) {
        self.match_formations += queries;
        self.classifier_tests += queries * population_size as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_episode_charges_environment_formations_at_the_start_size_and_learning_at_the_mean() {
        let mut cost = MatchingCost::default();
        cost.charge_episode(
            EpisodeFormations {
                environment: 4,
                learning: 6,
            },
            10,
            20,
        );
        assert_eq!(cost.match_formations, 10);
        assert_eq!(cost.classifier_tests, 4 * 10 + 6 * 15);
    }

    #[test]
    fn queries_are_charged_at_the_tested_population_size() {
        let mut cost = MatchingCost::default();
        cost.charge_queries(3, 7);
        cost.charge_queries(2, 5);
        assert_eq!(cost.match_formations, 5);
        assert_eq!(cost.classifier_tests, 31);
    }
}
