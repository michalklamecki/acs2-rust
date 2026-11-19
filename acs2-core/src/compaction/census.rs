use std::ops::Range;

use crate::compaction::cracs::reliable_mean_specificity;
use crate::compaction::evidence::{sample_value_carriers, states_of, StateMatches};
use crate::representation::population::Population;
use crate::support::knowledge::Transition;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ExperienceSummary {
    pub minimum: u32,
    pub lower_quartile: u32,
    pub median: u32,
    pub upper_quartile: u32,
    pub maximum: u32,
    pub mean: f64,
}

impl ExperienceSummary {
    pub fn of(mut experiences: Vec<u32>) -> Option<Self> {
        if experiences.is_empty() {
            return None;
        }
        experiences.sort_unstable();
        let last = experiences.len() - 1;
        let total: u64 = experiences.iter().map(|&value| u64::from(value)).sum();
        Some(Self {
            minimum: experiences[0],
            lower_quartile: experiences[last / 4],
            median: experiences[last / 2],
            upper_quartile: experiences[last * 3 / 4],
            maximum: experiences[last],
            mean: total as f64 / experiences.len() as f64,
        })
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Census {
    pub rules: usize,
    pub numerosity: u32,
    pub reliable: usize,
    pub marked: usize,
    pub change_anticipating: usize,
    pub goal_specific: usize,
    pub value_carriers: usize,
    pub experience: Option<ExperienceSummary>,
    pub mean_specificity: Option<f64>,
    pub reliable_mean_specificity: Option<f64>,
    pub work: u64,
}

pub fn population_census<const N: usize>(
    population: &Population<N>,
    theta_r: f64,
    goal_attributes: Range<usize>,
    transitions: &[Transition<N>],
) -> Census {
    let goal = goal_attributes.start..goal_attributes.end.min(N);
    let matches = StateMatches::collect(population, states_of(transitions));
    let carriers = sample_value_carriers(population, &matches, transitions);
    let specificity_total: usize = population
        .iter()
        .map(|classifier| classifier.condition.specificity())
        .sum();

    Census {
        rules: population.len(),
        numerosity: population.numerosity(),
        reliable: population.reliable_count(theta_r),
        marked: population.iter().filter(|classifier| classifier.is_marked()).count(),
        change_anticipating: population
            .iter()
            .filter(|classifier| classifier.does_anticipate_change())
            .count(),
        goal_specific: population
            .iter()
            .filter(|classifier| {
                goal.clone()
                    .any(|index| !classifier.condition.get(index).is_wildcard())
            })
            .count(),
        value_carriers: carriers.len(),
        experience: ExperienceSummary::of(population.iter().map(|classifier| classifier.exp).collect()),
        mean_specificity: (!population.is_empty())
            .then(|| specificity_total as f64 / (population.len() * N) as f64),
        reliable_mean_specificity: reliable_mean_specificity(population, theta_r),
        work: matches.work(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::test_support::{classifier, transition, THETA_R};
    use crate::representation::classifier::Classifier;
    use crate::representation::perception::Perception;
    use crate::representation::symbol::Symbol;

    fn rules() -> Vec<Classifier<3>> {
        let mut marked = classifier(b"1#0", 0, b"##1", 0.5);
        marked.set_mark(&Perception::new(b"110".map(Symbol::Token)));
        marked.exp = 2;
        let mut young = classifier(b"1##", 0, b"##1", 0.95);
        young.exp = 8;
        let mut old = classifier(b"###", 1, b"###", 0.95);
        old.exp = 100;
        let mut mid = classifier(b"00#", 1, b"1##", 0.3);
        mid.exp = 30;
        vec![marked, young, old, mid]
    }

    fn sample() -> Vec<Transition<3>> {
        vec![transition(b"100", 0, b"101"), transition(b"000", 1, b"100")]
    }

    #[test]
    fn counts_every_class_of_rule() {
        let population = Population::from_classifiers(rules());
        let census = population_census(&population, THETA_R, 2..3, &sample());
        assert_eq!(census.rules, 4);
        assert_eq!(census.numerosity, 4);
        assert_eq!(census.reliable, 2);
        assert_eq!(census.marked, 1);
        assert_eq!(census.change_anticipating, 3);
        assert_eq!(census.goal_specific, 1);
    }

    #[test]
    fn summarises_experience_and_specificity() {
        let population = Population::from_classifiers(rules());
        let census = population_census(&population, THETA_R, 0..0, &sample());
        let experience = census.experience.unwrap();
        assert_eq!(experience.minimum, 2);
        assert_eq!(experience.lower_quartile, 2);
        assert_eq!(experience.median, 8);
        assert_eq!(experience.upper_quartile, 30);
        assert_eq!(experience.maximum, 100);
        assert_eq!(experience.mean, 35.0);
        assert_eq!(census.mean_specificity, Some(5.0 / 12.0));
        assert_eq!(census.reliable_mean_specificity, Some(1.0 / 6.0));
        assert_eq!(census.goal_specific, 0);
    }

    #[test]
    fn goal_range_beyond_the_perception_is_clamped() {
        let population = Population::from_classifiers(rules());
        let census = population_census(&population, THETA_R, 2..99, &sample());
        assert_eq!(census.goal_specific, 1);
        let none = population_census(&population, THETA_R, 5..9, &sample());
        assert_eq!(none.goal_specific, 0);
    }

    #[test]
    fn value_carriers_are_counted_on_the_sample() {
        let mut population_rules = rules();
        population_rules[1].r = 1.0;
        population_rules[0].r = 0.5;
        population_rules[3].r = 0.7;
        let population = Population::from_classifiers(population_rules);
        let census = population_census(&population, THETA_R, 0..0, &sample());
        assert_eq!(census.value_carriers, 2);
        assert_eq!(census.work, 2 * 4);
        let without_sample = population_census(&population, THETA_R, 0..0, &[]);
        assert_eq!(without_sample.value_carriers, 0);
        assert_eq!(without_sample.work, 0);
    }

    #[test]
    fn empty_population_has_no_summaries() {
        let census = population_census(&Population::<3>::new(), THETA_R, 0..3, &sample());
        assert_eq!(census.rules, 0);
        assert_eq!(census.experience, None);
        assert_eq!(census.mean_specificity, None);
        assert_eq!(census.reliable_mean_specificity, None);
        assert_eq!(census.value_carriers, 0);
    }
}
