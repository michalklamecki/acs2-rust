use crate::compaction::cracs::{is_subsumed, CompactionConfiguration};
use crate::compaction::outcome::{ReductionOutcome, RemovalReason, Verdict};
use crate::compaction::rule::ValuedRule;
use crate::representation::classifier::Classifier;
use crate::representation::population::Population;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ValueGuard {
    tolerance: f64,
}

impl ValueGuard {
    pub fn new(tolerance: f64) -> Self {
        assert!(tolerance >= 0.0, "value tolerance must be non-negative");
        Self { tolerance }
    }

    pub fn exact() -> Self {
        Self::new(0.0)
    }

    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    pub fn permits<R: ValuedRule>(&self, candidate: &R, survivor: &R) -> bool {
        !candidate.anticipates_change()
            || (survivor.anticipates_change()
                && survivor.fitness() >= candidate.fitness() - self.tolerance)
    }
}

pub fn absorb<R, F>(rules: &[R], may_absorb: F, guard: Option<ValueGuard>) -> ReductionOutcome
where
    R: ValuedRule,
    F: Fn(&R, &R) -> bool,
{
    let mut order: Vec<usize> = (0..rules.len()).collect();
    order.sort_by_key(|&index| (rules[index].specificity(), index));

    let mut outcome = ReductionOutcome::keeping_all(rules.len(), 0);
    let mut survivors: Vec<usize> = Vec::new();
    for &candidate in &order {
        let mut eligible = false;
        let mut absorbed_by = None;
        for &survivor in &survivors {
            outcome.work += 1;
            if !may_absorb(&rules[candidate], &rules[survivor]) {
                continue;
            }
            eligible = true;
            let permitted =
                guard.is_none_or(|guard| guard.permits(&rules[candidate], &rules[survivor]));
            if permitted {
                absorbed_by = Some(survivor);
                break;
            }
        }
        match absorbed_by {
            Some(by) => {
                outcome.decisions[candidate] = Verdict::Removed(RemovalReason::Subsumed { by });
            }
            None => {
                if eligible {
                    outcome.value_vetoes += 1;
                }
                survivors.push(candidate);
            }
        }
    }
    outcome
}

pub fn compact_guarded<R: ValuedRule>(
    rules: &[R],
    config: &CompactionConfiguration,
    guard: Option<ValueGuard>,
) -> ReductionOutcome {
    absorb(
        rules,
        |candidate, survivor| is_subsumed(candidate, survivor, config),
        guard,
    )
}

pub fn compact_guarded_population<const N: usize>(
    population: &Population<N>,
    config: &CompactionConfiguration,
    guard: Option<ValueGuard>,
) -> (Population<N>, ReductionOutcome) {
    let outcome = compact_guarded::<Classifier<N>>(population.classifiers(), config, guard);
    (outcome.apply(population), outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::outcome::Verdict;
    use crate::compaction::rule::CompactableRule;
    use crate::compaction::test_support::{
        all_transitions, classifier, maximum_fitness_table, random_population, THETA_EXP, THETA_R,
    };
    use crate::support::knowledge::anticipation_fraction;
    use crate::support::rng::ChaChaRandomSource;

    fn config() -> CompactionConfiguration {
        CompactionConfiguration::new(THETA_EXP, THETA_R)
    }

    fn with_fitness(mut rule: Classifier<3>, fitness: f64) -> Classifier<3> {
        rule.q = 0.95;
        rule.r = fitness / 0.95;
        rule
    }

    #[derive(Clone)]
    struct LaxRule(Classifier<3>);

    impl CompactableRule for LaxRule {
        fn condition_includes(&self, other: &Self) -> bool {
            self.0.condition_includes(&other.0)
        }
        fn same_action(&self, other: &Self) -> bool {
            self.0.same_action(&other.0)
        }
        fn effect_includes(&self, _other: &Self) -> bool {
            true
        }
        fn quality(&self) -> f64 {
            self.0.q
        }
        fn experience(&self) -> u32 {
            self.0.exp
        }
        fn mark_permits(&self, other: &Self) -> bool {
            self.0.mark_permits(&other.0)
        }
        fn action_sequence_len(&self) -> usize {
            1
        }
        fn anticipates_change(&self) -> bool {
            self.0.anticipates_change()
        }
    }

    impl ValuedRule for LaxRule {
        fn fitness(&self) -> f64 {
            ValuedRule::fitness(&self.0)
        }
        fn specificity(&self) -> usize {
            ValuedRule::specificity(&self.0)
        }
        fn is_marked(&self) -> bool {
            ValuedRule::is_marked(&self.0)
        }
    }

    #[test]
    fn exact_guard_blocks_a_weaker_survivor() {
        let rules = vec![
            with_fitness(classifier(b"10#", 0, b"##1", 0.95), 0.9),
            with_fitness(classifier(b"1##", 0, b"##1", 0.95), 0.5),
        ];
        let exact = compact_guarded(&rules, &config(), Some(ValueGuard::exact()));
        assert_eq!(exact.kept_indices(), vec![0, 1]);
        assert_eq!(exact.value_vetoes, 1);

        let tolerant = compact_guarded(&rules, &config(), Some(ValueGuard::new(0.5)));
        assert_eq!(
            tolerant.decisions,
            vec![Verdict::Removed(RemovalReason::Subsumed { by: 1 }), Verdict::Kept]
        );
        assert_eq!(tolerant.value_vetoes, 0);

        let unguarded = compact_guarded(&rules, &config(), None);
        assert_eq!(unguarded.kept_indices(), vec![1]);
    }

    #[test]
    fn stronger_or_equal_survivor_absorbs_under_exact_guard() {
        let rules = vec![
            with_fitness(classifier(b"10#", 0, b"##1", 0.95), 0.5),
            with_fitness(classifier(b"1##", 0, b"##1", 0.95), 0.5),
        ];
        let outcome = compact_guarded(&rules, &config(), Some(ValueGuard::exact()));
        assert_eq!(outcome.kept_indices(), vec![1]);
    }

    #[test]
    fn tolerance_does_not_accumulate_along_a_chain() {
        let rules = vec![
            with_fitness(classifier(b"101", 0, b"##1", 0.95), 0.70),
            with_fitness(classifier(b"10#", 0, b"##1", 0.95), 0.65),
            with_fitness(classifier(b"1##", 0, b"##1", 0.95), 0.60),
        ];
        let outcome = compact_guarded(&rules, &config(), Some(ValueGuard::new(0.06)));
        assert_eq!(
            outcome.decisions,
            vec![
                Verdict::Kept,
                Verdict::Removed(RemovalReason::Subsumed { by: 2 }),
                Verdict::Kept,
            ]
        );
        assert_eq!(outcome.value_vetoes, 1);
    }

    #[test]
    fn survivors_are_always_kept_rules() {
        let rules = vec![
            classifier(b"101", 0, b"##1", 0.95),
            classifier(b"10#", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"###", 0, b"##1", 0.95),
        ];
        let outcome = compact_guarded(&rules, &config(), Some(ValueGuard::exact()));
        assert_eq!(outcome.kept_indices(), vec![3]);
        for verdict in &outcome.decisions[..3] {
            assert_eq!(*verdict, Verdict::Removed(RemovalReason::Subsumed { by: 3 }));
        }
    }

    #[test]
    fn identical_rules_keep_the_lower_index() {
        let rules = vec![
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
        ];
        let outcome = compact_guarded(&rules, &config(), Some(ValueGuard::exact()));
        assert_eq!(
            outcome.decisions,
            vec![Verdict::Kept, Verdict::Removed(RemovalReason::Subsumed { by: 0 })]
        );
    }

    #[test]
    fn rules_without_anticipated_change_carry_no_value_and_pass_the_guard() {
        let rules = vec![
            with_fitness(classifier(b"10#", 0, b"###", 0.95), 0.9),
            with_fitness(classifier(b"1##", 0, b"###", 0.95), 0.0),
        ];
        let outcome = compact_guarded(&rules, &config(), Some(ValueGuard::exact()));
        assert_eq!(outcome.kept_indices(), vec![1]);
    }

    #[test]
    fn change_anticipating_rule_needs_a_change_anticipating_survivor() {
        let rules = vec![
            LaxRule(with_fitness(classifier(b"10#", 0, b"##1", 0.95), 0.1)),
            LaxRule(with_fitness(classifier(b"1##", 0, b"###", 0.95), 0.9)),
        ];
        let guarded = compact_guarded(&rules, &config(), Some(ValueGuard::new(10.0)));
        assert_eq!(guarded.kept_indices(), vec![0, 1]);
        assert_eq!(guarded.value_vetoes, 1);

        let unguarded = compact_guarded(&rules, &config(), None);
        assert_eq!(unguarded.kept_indices(), vec![1]);

        let reversed = vec![
            LaxRule(with_fitness(classifier(b"10#", 0, b"###", 0.95), 0.1)),
            LaxRule(with_fitness(classifier(b"1##", 0, b"##1", 0.95), 0.9)),
        ];
        let allowed = compact_guarded(&reversed, &config(), Some(ValueGuard::exact()));
        assert_eq!(allowed.kept_indices(), vec![1]);
    }

    #[test]
    fn work_counts_every_candidate_against_every_earlier_kept_rule() {
        let rules = vec![
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"0##", 0, b"##1", 0.95),
            classifier(b"00#", 1, b"##1", 0.95),
        ];
        let outcome = compact_guarded(&rules, &config(), None);
        assert_eq!(outcome.kept_indices(), vec![0, 1, 2]);
        assert_eq!(outcome.work, 3);
    }

    #[test]
    fn empty_population_is_a_no_op() {
        let outcome = compact_guarded::<Classifier<3>>(&[], &config(), Some(ValueGuard::exact()));
        assert!(outcome.decisions.is_empty());
        assert_eq!(outcome.work, 0);
        assert_eq!(outcome.compression_rate(), 0.0);
    }

    #[test]
    #[should_panic]
    fn negative_tolerance_is_rejected() {
        ValueGuard::new(-0.1);
    }

    fn knowledge(population: &Population<3>) -> f64 {
        anticipation_fraction(population, THETA_R, all_transitions())
    }

    fn assert_pointers_are_final_survivors(population: &Population<3>, outcome: &ReductionOutcome) {
        for (index, verdict) in outcome.decisions.iter().enumerate() {
            if let Verdict::Removed(RemovalReason::Subsumed { by }) = verdict {
                assert_eq!(outcome.decisions[*by], Verdict::Kept);
                let removed = population.get(index);
                let survivor = population.get(*by);
                assert!(survivor.condition.includes(&removed.condition));
                assert_eq!(survivor.action, removed.action);
                assert_eq!(survivor.effect, removed.effect);
                if removed.is_reliable(THETA_R) {
                    assert!(survivor.is_reliable(THETA_R));
                }
            }
        }
    }

    #[test]
    fn exact_guard_on_cracs_keeps_knowledge_and_every_value() {
        let mut rng = ChaChaRandomSource::from_seed(11);
        let mut removed_total = 0;
        for _ in 0..200 {
            let population = random_population(&mut rng, 30);
            let (reduced, outcome) =
                compact_guarded_population(&population, &config(), Some(ValueGuard::exact()));
            removed_total += outcome.removed_count();

            assert_eq!(knowledge(&population), knowledge(&reduced));
            assert_eq!(maximum_fitness_table(&population), maximum_fitness_table(&reduced));
            assert_pointers_are_final_survivors(&population, &outcome);

            let (again, second) =
                compact_guarded_population(&reduced, &config(), Some(ValueGuard::exact()));
            assert_eq!(second.removed_count(), 0);
            assert_eq!(again.len(), reduced.len());

            let (repeat, repeat_outcome) =
                compact_guarded_population(&population, &config(), Some(ValueGuard::exact()));
            assert_eq!(repeat_outcome, outcome);
            assert_eq!(repeat.classifiers(), reduced.classifiers());
        }
        assert!(removed_total > 0);
    }

    #[test]
    fn tolerance_bounds_the_value_change_in_every_state_and_action() {
        let mut rng = ChaChaRandomSource::from_seed(12);
        let tolerance = 0.1;
        for _ in 0..200 {
            let population = random_population(&mut rng, 30);
            let (reduced, _) = compact_guarded_population(
                &population,
                &config(),
                Some(ValueGuard::new(tolerance)),
            );
            let before = maximum_fitness_table(&population);
            let after = maximum_fitness_table(&reduced);
            for (value_before, value_after) in before.iter().zip(&after) {
                assert!(*value_after <= *value_before);
                assert!(*value_before - *value_after <= tolerance);
            }
            assert_eq!(knowledge(&population), knowledge(&reduced));
        }
    }
}
