use crate::compaction::controller::{
    no_candidate_check, CompactionController, CompactionMode, CompactionParameters,
    CompactionSignals, ReductionRecord, TaskGeometry, COMPACTION_STREAM,
};
use crate::compaction::cost::{EpisodeFormations, MatchingCost};
use crate::compaction::evidence_source::EvidenceSource;
use crate::representation::population::Population;
use crate::support::rng::{ChaChaRandomSource, RandomSource};

#[derive(Clone, Debug)]
pub struct SingleHeadCompaction<const N: usize, C> {
    controller: CompactionController<N>,
    cost: MatchingCost,
    rng: C,
}

impl<const N: usize> SingleHeadCompaction<N, ChaChaRandomSource> {
    pub fn seeded(parameters: CompactionParameters, geometry: TaskGeometry, seed: u64) -> Self {
        Self::new(
            parameters,
            geometry,
            ChaChaRandomSource::from_seed_and_stream(seed, COMPACTION_STREAM),
        )
    }
}

impl<const N: usize, C: RandomSource> SingleHeadCompaction<N, C> {
    pub fn new(parameters: CompactionParameters, geometry: TaskGeometry, rng: C) -> Self {
        Self {
            controller: CompactionController::new(parameters, geometry),
            cost: MatchingCost::default(),
            rng,
        }
    }

    pub fn mode(&self) -> CompactionMode {
        self.controller.mode()
    }

    pub fn is_active(&self) -> bool {
        self.controller.mode() != CompactionMode::Off
    }

    pub fn controller(&self) -> &CompactionController<N> {
        &self.controller
    }

    pub fn records(&self) -> &[ReductionRecord] {
        self.controller.records()
    }

    pub fn cost(&self) -> MatchingCost {
        self.cost
    }

    pub fn random_source(&self) -> &C {
        &self.rng
    }

    pub fn charge_episode(
        &mut self,
        formations: EpisodeFormations,
        size_at_start: usize,
        size_at_end: usize,
    ) {
        self.cost
            .charge_episode(formations, size_at_start, size_at_end);
    }

    pub fn reduce<E>(
        &mut self,
        population: &Population<N>,
        interactions: u64,
        evidence: &E,
    ) -> Option<Population<N>>
    where
        E: EvidenceSource<N> + ?Sized,
    {
        let signals = CompactionSignals {
            interactions,
            population_size: population.len(),
            match_formations: self.cost.match_formations,
            classifier_tests: self.cost.classifier_tests,
            rules_created: None,
            coverage_permille: None,
        };
        self.controller
            .step(
                population,
                &signals,
                evidence,
                &mut self.rng,
                &mut no_candidate_check,
            )
            .and_then(|report| report.replacement())
    }

    pub fn snapshot(&self) -> String {
        format!(
            "{:?}|{:?}|{:?}",
            self.controller,
            self.cost,
            self.rng.capture_state()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::test_support::{classifier, ACTIONS, THETA_EXP, THETA_R};
    use crate::compaction::trigger::Periodic;
    use crate::compaction::{
        fixed_strength_policy, CompactionStrength, NoEvidence, Operator,
    };

    fn parameters(mode: CompactionMode) -> CompactionParameters {
        let strength = CompactionStrength {
            value_tolerance: 0.0,
            maturity_experience: THETA_EXP,
            idle_audits: 1,
        };
        let policy = fixed_strength_policy(Periodic::new(5).unwrap(), strength).unwrap();
        CompactionParameters::builder(THETA_R, policy)
            .mode(mode)
            .operator(Operator::MatureSubsumption { guarded: true })
            .build()
            .unwrap()
    }

    fn geometry() -> TaskGeometry {
        TaskGeometry {
            actions: ACTIONS,
            goal_attributes: 2..3,
        }
    }

    fn redundant_pair() -> Population<3> {
        Population::from_classifiers(vec![
            classifier(b"10#", 0, b"##1", 0.5),
            classifier(b"1##", 0, b"##1", 0.95),
        ])
    }

    #[test]
    fn it_charges_the_derived_cost_and_feeds_it_to_the_trigger() {
        let mut compaction = SingleHeadCompaction::seeded(parameters(CompactionMode::Shadow), geometry(), 3);
        compaction.charge_episode(
            EpisodeFormations {
                environment: 3,
                learning: 9,
            },
            4,
            6,
        );
        assert_eq!(compaction.cost().match_formations, 12);
        assert_eq!(compaction.cost().classifier_tests, 3 * 4 + 9 * 5);
        let population = redundant_pair();
        assert!(compaction.reduce(&population, 2, &NoEvidence).is_none());
        assert!(compaction.reduce(&population, 6, &NoEvidence).is_none());
        let record = compaction.records().last().unwrap();
        assert_eq!(record.observation.match_formations_since_reduction, 12);
        assert_eq!(record.observation.classifier_tests_since_reduction, 57);
        assert_eq!(record.interaction, 6);
    }

    #[test]
    fn apply_returns_the_replacement_and_shadow_never_does() {
        let population = redundant_pair();
        let mut shadow = SingleHeadCompaction::seeded(parameters(CompactionMode::Shadow), geometry(), 3);
        assert!(shadow.reduce(&population, 5, &NoEvidence).is_none());
        assert_eq!(shadow.records().len(), 1);
        assert!(!shadow.records()[0].applied);
        let mut apply = SingleHeadCompaction::seeded(parameters(CompactionMode::Apply), geometry(), 3);
        let replacement = apply.reduce(&population, 5, &NoEvidence);
        assert!(apply.records()[0].removals.total() > 0 || replacement.is_none());
    }

    #[test]
    fn an_off_mode_never_reduces_nor_draws_randomness() {
        let mut off = SingleHeadCompaction::seeded(CompactionParameters::off(), geometry(), 3);
        assert!(!off.is_active());
        assert!(off.reduce(&redundant_pair(), 100, &NoEvidence).is_none());
        assert!(off.records().is_empty());
        assert_eq!(off.random_source().capture_state().unwrap().word_pos, 0);
    }
}
