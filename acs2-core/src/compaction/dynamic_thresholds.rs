use crate::compaction::outcome::{ReductionOutcome, RemovalReason, Verdict};
use crate::representation::classifier::Classifier;
use crate::representation::population::Population;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DynamicThresholds {
    pub size_threshold: usize,
    pub experience_threshold: u32,
    pub quality_threshold: f64,
}

impl DynamicThresholds {
    pub fn new(size_threshold: usize, experience_threshold: u32, quality_threshold: f64) -> Self {
        Self {
            size_threshold,
            experience_threshold,
            quality_threshold,
        }
    }

    pub fn is_removable<const N: usize>(&self, classifier: &Classifier<N>) -> bool {
        classifier.exp >= self.experience_threshold && classifier.q < self.quality_threshold
    }
}

pub fn prune_by_thresholds<const N: usize>(
    rules: &[Classifier<N>],
    thresholds: &DynamicThresholds,
) -> ReductionOutcome {
    let mut outcome = ReductionOutcome::keeping_all(rules.len(), 0);
    if rules.len() <= thresholds.size_threshold {
        return outcome;
    }
    for (index, rule) in rules.iter().enumerate() {
        if thresholds.is_removable(rule) {
            outcome.decisions[index] = Verdict::Removed(RemovalReason::LowQuality);
        }
    }
    outcome
}

pub fn prune_by_thresholds_population<const N: usize>(
    population: &Population<N>,
    thresholds: &DynamicThresholds,
) -> (Population<N>, ReductionOutcome) {
    let outcome = prune_by_thresholds(population.classifiers(), thresholds);
    (outcome.apply(population), outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::outcome::RemovalReason;
    use crate::compaction::test_support::{classifier, random_population};
    use crate::support::rng::ChaChaRandomSource;

    fn rule(q: f64, exp: u32) -> Classifier<3> {
        let mut rule = classifier(b"1##", 0, b"##1", q);
        rule.exp = exp;
        rule
    }

    fn rules() -> Vec<Classifier<3>> {
        vec![rule(0.2, 40), rule(0.2, 3), rule(0.95, 40), rule(0.49, 25), rule(0.5, 25)]
    }

    #[test]
    fn nothing_is_removed_until_the_population_exceeds_the_size_threshold() {
        let thresholds = DynamicThresholds::new(5, 20, 0.5);
        let outcome = prune_by_thresholds(&rules(), &thresholds);
        assert_eq!(outcome.removed_count(), 0);
        assert_eq!(outcome.work, 0);
    }

    #[test]
    fn only_experienced_low_quality_rules_are_removed() {
        let thresholds = DynamicThresholds::new(4, 20, 0.5);
        let outcome = prune_by_thresholds(&rules(), &thresholds);
        assert_eq!(outcome.removed_indices(), vec![0, 3]);
        assert_eq!(outcome.decisions[0], Verdict::Removed(RemovalReason::LowQuality));
    }

    #[test]
    fn rules_still_learning_survive_regardless_of_quality() {
        let thresholds = DynamicThresholds::new(0, 20, 0.5);
        let outcome = prune_by_thresholds(&rules(), &thresholds);
        assert_eq!(outcome.decisions[1], Verdict::Kept);
    }

    #[test]
    fn experience_and_quality_boundaries_are_inclusive_and_exclusive_respectively() {
        let thresholds = DynamicThresholds::new(0, 25, 0.5);
        let outcome = prune_by_thresholds(&[rule(0.49, 25), rule(0.5, 25), rule(0.49, 24)], &thresholds);
        assert_eq!(outcome.removed_indices(), vec![0]);
    }

    #[test]
    fn population_helper_returns_the_pruned_copy() {
        let population = Population::from_classifiers(rules());
        let (pruned, outcome) =
            prune_by_thresholds_population(&population, &DynamicThresholds::new(4, 20, 0.5));
        assert_eq!(population.len(), 5);
        assert_eq!(pruned.len(), 3);
        assert_eq!(outcome.removed_count(), 2);
    }

    #[test]
    fn random_populations_lose_only_rules_that_pass_both_filters() {
        let mut rng = ChaChaRandomSource::from_seed(41);
        let thresholds = DynamicThresholds::new(10, 20, 0.6);
        for _ in 0..100 {
            let population = random_population(&mut rng, 30);
            let (pruned, outcome) = prune_by_thresholds_population(&population, &thresholds);
            for index in outcome.removed_indices() {
                assert!(thresholds.is_removable(population.get(index)));
            }
            assert!(pruned.iter().all(|rule| !thresholds.is_removable(rule))
                || population.len() <= thresholds.size_threshold);
            let (_, repeat) = prune_by_thresholds_population(&population, &thresholds);
            assert_eq!(repeat, outcome);
        }
    }
}
