use std::cmp::Ordering;
use std::collections::BTreeSet;

use crate::compaction::evidence::{sample_value_carriers, states_of, StateMatches};
use crate::compaction::outcome::{ReductionOutcome, RemovalReason, Verdict};
use crate::representation::population::Population;
use crate::support::knowledge::Transition;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CapacityConfiguration {
    pub capacity: usize,
}

impl CapacityConfiguration {
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }
}

pub fn evict_to_capacity<const N: usize>(
    population: &Population<N>,
    transitions: &[Transition<N>],
    config: &CapacityConfiguration,
) -> ReductionOutcome {
    let excess = population.len().saturating_sub(config.capacity);
    if excess == 0 {
        return ReductionOutcome::keeping_all(population.len(), 0);
    }
    let matches = StateMatches::collect(population, states_of(transitions));
    let carriers = sample_value_carriers(population, &matches, transitions);
    let mut order: Vec<usize> = (0..population.len()).collect();
    order.sort_by(|&left, &right| eviction_order(population, &carriers, left, right));
    let mut outcome = ReductionOutcome::keeping_all(population.len(), matches.work());
    for &index in order.iter().take(excess) {
        outcome.decisions[index] = Verdict::Removed(RemovalReason::Evicted);
    }
    outcome
}

pub fn evict_to_capacity_population<const N: usize>(
    population: &Population<N>,
    transitions: &[Transition<N>],
    config: &CapacityConfiguration,
) -> (Population<N>, ReductionOutcome) {
    let outcome = evict_to_capacity(population, transitions, config);
    (outcome.apply(population), outcome)
}

fn eviction_order<const N: usize>(
    population: &Population<N>,
    carriers: &BTreeSet<usize>,
    left: usize,
    right: usize,
) -> Ordering {
    let (a, b) = (population.get(left), population.get(right));
    carriers
        .contains(&left)
        .cmp(&carriers.contains(&right))
        .then(a.q.total_cmp(&b.q))
        .then(a.exp.cmp(&b.exp))
        .then(right.cmp(&left))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::test_support::{classifier, random_population, transition};
    use crate::representation::classifier::Classifier;
    use crate::support::rng::{ChaChaRandomSource, RandomSource};

    fn sample() -> Vec<Transition<3>> {
        vec![transition(b"100", 0, b"101")]
    }

    fn rule(condition: &[u8; 3], action: usize, q: f64, exp: u32) -> Classifier<3> {
        let mut rule = classifier(condition, action, b"##1", q);
        rule.exp = exp;
        rule
    }

    fn evicted(population: &Population<3>, capacity: usize) -> Vec<usize> {
        evict_to_capacity(population, &sample(), &CapacityConfiguration::new(capacity))
            .removed_indices()
    }

    #[test]
    fn population_within_capacity_is_untouched() {
        let population = Population::from_classifiers(vec![rule(b"1##", 0, 0.9, 5)]);
        let outcome = evict_to_capacity(&population, &sample(), &CapacityConfiguration::new(1));
        assert_eq!(outcome.removed_count(), 0);
        assert_eq!(outcome.work, 0);
    }

    #[test]
    fn lowest_quality_goes_first() {
        let population = Population::from_classifiers(vec![
            rule(b"0##", 1, 0.5, 9),
            rule(b"0##", 1, 0.2, 9),
            rule(b"0##", 1, 0.8, 9),
        ]);
        assert_eq!(evicted(&population, 2), vec![1]);
        assert_eq!(evicted(&population, 1), vec![0, 1]);
    }

    #[test]
    fn lower_experience_goes_before_higher_at_equal_quality() {
        let population = Population::from_classifiers(vec![
            rule(b"0##", 1, 0.5, 30),
            rule(b"0##", 1, 0.5, 4),
        ]);
        assert_eq!(evicted(&population, 1), vec![1]);
    }

    #[test]
    fn highest_index_goes_first_when_everything_ties() {
        let population = Population::from_classifiers(vec![
            rule(b"0##", 1, 0.5, 9),
            rule(b"0##", 1, 0.5, 9),
            rule(b"0##", 1, 0.5, 9),
        ]);
        assert_eq!(evicted(&population, 1), vec![1, 2]);
    }

    #[test]
    fn value_carriers_go_after_every_non_carrier() {
        let mut carrier = rule(b"1##", 0, 0.1, 1);
        carrier.r = 5.0;
        let population = Population::from_classifiers(vec![
            carrier,
            rule(b"0##", 1, 0.95, 90),
            rule(b"0##", 1, 0.95, 90),
        ]);
        assert_eq!(evicted(&population, 1), vec![1, 2]);
        assert_eq!(evicted(&population, 2), vec![2]);
    }

    #[test]
    fn zero_capacity_evicts_everything() {
        let population = Population::from_classifiers(vec![
            rule(b"0##", 1, 0.5, 9),
            rule(b"1##", 0, 0.5, 9),
        ]);
        assert_eq!(evicted(&population, 0), vec![0, 1]);
    }

    #[test]
    fn work_is_distinct_sample_states_times_population_size() {
        let population = Population::from_classifiers(vec![
            rule(b"0##", 1, 0.5, 9),
            rule(b"1##", 0, 0.5, 9),
            rule(b"1##", 1, 0.5, 9),
        ]);
        let mut transitions = sample();
        transitions.push(transition(b"100", 1, b"100"));
        transitions.push(transition(b"011", 1, b"111"));
        let outcome =
            evict_to_capacity(&population, &transitions, &CapacityConfiguration::new(1));
        assert_eq!(outcome.work, 2 * 3);
    }

    #[test]
    fn population_helper_enforces_the_capacity() {
        let mut rng = ChaChaRandomSource::from_seed(51);
        for _ in 0..100 {
            let population = random_population(&mut rng, 30);
            let capacity = rng.gen_range(20);
            let config = CapacityConfiguration::new(capacity);
            let (reduced, outcome) = evict_to_capacity_population(&population, &sample(), &config);
            assert_eq!(reduced.len(), population.len().min(capacity));
            assert!(outcome
                .decisions
                .iter()
                .all(|verdict| matches!(verdict, Verdict::Kept | Verdict::Removed(RemovalReason::Evicted))));
            let (_, repeat) = evict_to_capacity_population(&population, &sample(), &config);
            assert_eq!(repeat, outcome);
        }
    }
}
