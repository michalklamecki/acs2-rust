use std::collections::{BTreeMap, BTreeSet};

use crate::compaction::evidence::{
    best_reliable_anticipator, sample_value_carriers, states_of, StateMatches,
};
use crate::compaction::outcome::{ReductionOutcome, RemovalReason, Verdict};
use crate::compaction::rule_key::RuleKey;
use crate::representation::population::{ClassifierRef, Population};
use crate::support::config::Configuration;
use crate::support::knowledge::Transition;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IdlePruningConfiguration {
    pub theta_r: f64,
    pub required_audits: u32,
    pub min_experience: u32,
}

impl IdlePruningConfiguration {
    pub fn new(theta_r: f64, required_audits: u32, min_experience: u32) -> Self {
        Self {
            theta_r,
            required_audits,
            min_experience,
        }
    }

    pub fn from_learning(config: &Configuration, required_audits: u32) -> Self {
        Self::new(config.theta_r, required_audits, config.theta_exp)
    }
}

pub fn idle_flags<const N: usize>(
    population: &Population<N>,
    transitions: &[Transition<N>],
    config: &IdlePruningConfiguration,
) -> (Vec<bool>, u64) {
    let matches = StateMatches::collect(population, states_of(transitions));
    let mut protected: BTreeSet<ClassifierRef> = BTreeSet::new();
    for transition in transitions {
        let action_set = matches.action_set(population, &transition.p0, transition.action);
        match best_reliable_anticipator(population, &action_set, transition, config.theta_r) {
            Some(best) => {
                protected.insert(best);
            }
            None => protected.extend(action_set),
        }
    }
    protected.extend(sample_value_carriers(population, &matches, transitions));
    let flags = (0..population.len())
        .map(|reference| {
            !protected.contains(&reference)
                && population.get(reference).exp >= config.min_experience
        })
        .collect();
    (flags, matches.work())
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IdleTracker<const N: usize> {
    streaks: BTreeMap<RuleKey<N>, u32>,
}

impl<const N: usize> Default for IdleTracker<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> IdleTracker<N> {
    pub fn new() -> Self {
        Self {
            streaks: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.streaks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.streaks.is_empty()
    }

    pub fn streak(&self, key: &RuleKey<N>) -> u32 {
        self.streaks.get(key).copied().unwrap_or(0)
    }

    pub fn observe(
        &self,
        population: &Population<N>,
        transitions: &[Transition<N>],
        config: &IdlePruningConfiguration,
    ) -> (ReductionOutcome, IdleTracker<N>) {
        if transitions.is_empty() {
            return (
                ReductionOutcome::keeping_all(population.len(), 0),
                self.carried_over(population),
            );
        }
        let (flags, work) = idle_flags(population, transitions, config);
        let keys: Vec<RuleKey<N>> = population.iter().map(RuleKey::of).collect();
        let mut key_idle: BTreeMap<&RuleKey<N>, bool> = BTreeMap::new();
        for (key, idle) in keys.iter().zip(&flags) {
            key_idle
                .entry(key)
                .and_modify(|all_idle| *all_idle &= *idle)
                .or_insert(*idle);
        }
        let required = config.required_audits.max(1);
        let mut outcome = ReductionOutcome::keeping_all(population.len(), work);
        let mut next = BTreeMap::new();
        for (index, key) in keys.iter().enumerate() {
            if !key_idle[key] {
                continue;
            }
            let streak = self.streak(key) + 1;
            if streak >= required {
                outcome.decisions[index] = Verdict::Removed(RemovalReason::Idle { audits: streak });
            } else {
                next.insert(key.clone(), streak);
            }
        }
        (outcome, IdleTracker { streaks: next })
    }

    pub fn observe_population(
        &self,
        population: &Population<N>,
        transitions: &[Transition<N>],
        config: &IdlePruningConfiguration,
    ) -> (Population<N>, ReductionOutcome, IdleTracker<N>) {
        let (outcome, tracker) = self.observe(population, transitions, config);
        (outcome.apply(population), outcome, tracker)
    }

    fn carried_over(&self, population: &Population<N>) -> IdleTracker<N> {
        let present: BTreeSet<RuleKey<N>> = population.iter().map(RuleKey::of).collect();
        IdleTracker {
            streaks: self
                .streaks
                .iter()
                .filter(|(key, _)| present.contains(*key))
                .map(|(key, streak)| (key.clone(), *streak))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::test_support::{
        classifier, environment_transitions, grounded_population, maximum_fitness_table, transition,
        THETA_R,
    };
    use crate::representation::classifier::Classifier;
    use crate::support::knowledge::anticipation_fraction;
    use crate::support::rng::ChaChaRandomSource;

    fn config(required_audits: u32) -> IdlePruningConfiguration {
        IdlePruningConfiguration::new(THETA_R, required_audits, 0)
    }

    fn key(rule: &Classifier<3>) -> RuleKey<3> {
        RuleKey::of(rule)
    }

    fn sample() -> Vec<Transition<3>> {
        vec![
            transition(b"100", 0, b"101"),
            transition(b"000", 0, b"000"),
            transition(b"111", 1, b"011"),
        ]
    }

    fn scenario() -> Vec<Classifier<3>> {
        vec![
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"10#", 0, b"##1", 0.93),
            classifier(b"0##", 0, b"###", 0.95),
            classifier(b"#0#", 1, b"###", 0.2),
            classifier(b"11#", 1, b"0##", 0.3),
            classifier(b"0##", 1, b"1##", 0.3),
        ]
    }

    fn idle_of(rules: Vec<Classifier<3>>, transitions: &[Transition<3>]) -> Vec<bool> {
        idle_flags(&Population::from_classifiers(rules), transitions, &config(1)).0
    }

    #[test]
    fn best_anticipator_bump_cover_and_uncovered_matches_are_protected() {
        let flags = idle_of(scenario(), &sample());
        assert_eq!(flags, vec![false, true, false, true, false, true]);
    }

    #[test]
    fn best_anticipator_ties_go_to_the_lowest_index() {
        let mut rules = vec![
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"10#", 0, b"##1", 0.95),
        ];
        rules[1].r = 0.5;
        assert_eq!(idle_of(rules, &sample()), vec![false, true]);
    }

    #[test]
    fn value_carrier_is_protected_even_when_it_is_not_the_best_anticipator() {
        let mut rules = scenario();
        let mut carrier = classifier(b"10#", 0, b"##1", 0.91);
        carrier.r = 2.0;
        rules[1] = carrier;
        assert!(!idle_of(rules, &sample())[1]);
    }

    #[test]
    fn zero_fitness_rules_are_not_value_carriers() {
        let mut rules = scenario();
        rules[0].r = 0.0;
        rules[1].r = 0.0;
        let flags = idle_of(rules, &sample());
        assert!(!flags[0]);
        assert!(flags[1]);
    }

    #[test]
    fn young_rules_are_never_idle() {
        let mut rules = scenario();
        rules[1].exp = 3;
        let population = Population::from_classifiers(rules);
        let strict = IdlePruningConfiguration::new(THETA_R, 1, 10);
        let (flags, _) = idle_flags(&population, &sample(), &strict);
        assert!(!flags[1]);
        assert!(flags[3]);
    }

    #[test]
    fn removal_needs_k_consecutive_idle_audits() {
        let population = Population::from_classifiers(scenario());
        let transitions = sample();
        let mut tracker = IdleTracker::new();
        for expected_streak in 1..3 {
            let (outcome, next) = tracker.observe(&population, &transitions, &config(3));
            assert_eq!(outcome.removed_count(), 0);
            assert_eq!(next.streak(&key(population.get(1))), expected_streak);
            tracker = next;
        }
        let (outcome, next) = tracker.observe(&population, &transitions, &config(3));
        assert_eq!(outcome.removed_indices(), vec![1, 3, 5]);
        assert_eq!(
            outcome.decisions[1],
            Verdict::Removed(RemovalReason::Idle { audits: 3 })
        );
        assert!(next.is_empty());
    }

    #[test]
    fn a_protected_audit_resets_the_streak() {
        let full = Population::from_classifiers(scenario());
        let mut without_best = scenario();
        without_best.remove(0);
        let reduced = Population::from_classifiers(without_best);
        let transitions = sample();
        let target = key(full.get(1));

        let (_, after_one) = IdleTracker::new().observe(&full, &transitions, &config(2));
        assert_eq!(after_one.streak(&target), 1);
        let (outcome, after_reset) = after_one.observe(&reduced, &transitions, &config(2));
        assert_eq!(outcome.decisions[0], Verdict::Kept);
        assert_eq!(after_reset.streak(&target), 0);
        let (outcome, _) = after_reset.observe(&full, &transitions, &config(2));
        assert_eq!(outcome.decisions[1], Verdict::Kept);
    }

    #[test]
    fn rules_absent_from_the_population_are_forgotten() {
        let full = Population::from_classifiers(scenario());
        let (_, tracker) = IdleTracker::new().observe(&full, &sample(), &config(5));
        assert_eq!(tracker.len(), 3);
        let only_first = Population::from_classifiers(vec![scenario().remove(0)]);
        let (_, next) = tracker.observe(&only_first, &sample(), &config(5));
        assert!(next.is_empty());
        let (_, carried) = tracker.observe(&only_first, &[], &config(5));
        assert!(carried.is_empty());
    }

    #[test]
    fn empty_sample_is_no_evidence() {
        let population = Population::from_classifiers(scenario());
        let (_, tracker) = IdleTracker::new().observe(&population, &sample(), &config(5));
        let (outcome, next) = tracker.observe(&population, &[], &config(1));
        assert_eq!(outcome.removed_count(), 0);
        assert_eq!(outcome.work, 0);
        assert_eq!(next, tracker);
    }

    #[test]
    fn one_audit_suffices_when_required_audits_is_zero_or_one() {
        let population = Population::from_classifiers(scenario());
        for required in [0, 1] {
            let (outcome, tracker) =
                IdleTracker::new().observe(&population, &sample(), &config(required));
            assert_eq!(outcome.removed_indices(), vec![1, 3, 5]);
            assert!(tracker.is_empty());
        }
    }

    #[test]
    fn duplicate_structure_is_removed_only_when_every_copy_is_idle() {
        let rules = vec![
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"#0#", 1, b"###", 0.2),
            classifier(b"#0#", 1, b"###", 0.2),
        ];
        let population = Population::from_classifiers(rules);
        let (outcome, _) = IdleTracker::new().observe(&population, &sample(), &config(1));
        assert_eq!(outcome.removed_indices(), vec![2, 3]);
    }

    #[test]
    fn work_is_distinct_sample_states_times_population_size() {
        let population = Population::from_classifiers(scenario());
        let mut transitions = sample();
        transitions.push(transition(b"100", 1, b"100"));
        let (outcome, _) = IdleTracker::new().observe(&population, &transitions, &config(1));
        assert_eq!(outcome.work, 3 * 6);
    }

    #[test]
    fn population_helper_returns_reduced_population_and_next_tracker() {
        let population = Population::from_classifiers(scenario());
        let (reduced, outcome, tracker) =
            IdleTracker::new().observe_population(&population, &sample(), &config(1));
        assert_eq!(population.len(), 6);
        assert_eq!(reduced.len(), 3);
        assert_eq!(outcome.removed_count(), 3);
        assert!(tracker.is_empty());
    }

    #[test]
    fn from_learning_uses_the_acs2_thresholds() {
        let learning = Configuration::default_protocol();
        let derived = IdlePruningConfiguration::from_learning(&learning, 4);
        assert_eq!(derived.theta_r, learning.theta_r);
        assert_eq!(derived.min_experience, learning.theta_exp);
        assert_eq!(derived.required_audits, 4);
    }

    #[test]
    fn single_audit_keeps_sample_knowledge_and_every_value_on_random_populations() {
        let mut rng = ChaChaRandomSource::from_seed(31);
        let transitions_of = environment_transitions;
        let mut removed_total = 0;
        for _ in 0..200 {
            let population = grounded_population(&mut rng, 30);
            let (reduced, outcome, _) =
                IdleTracker::new().observe_population(&population, &transitions_of(), &config(1));
            removed_total += outcome.removed_count();
            assert_eq!(
                anticipation_fraction(&population, THETA_R, transitions_of()),
                anticipation_fraction(&reduced, THETA_R, transitions_of())
            );
            assert_eq!(maximum_fitness_table(&population), maximum_fitness_table(&reduced));

            let (_, second, _) =
                IdleTracker::new().observe_population(&reduced, &transitions_of(), &config(1));
            assert_eq!(second.removed_count(), 0);

            let (_, repeat, _) =
                IdleTracker::new().observe_population(&population, &transitions_of(), &config(1));
            assert_eq!(repeat, outcome);
        }
        assert!(removed_total > 0);
    }
}
