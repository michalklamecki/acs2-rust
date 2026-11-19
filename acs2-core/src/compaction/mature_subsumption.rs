use crate::compaction::outcome::ReductionOutcome;
use crate::compaction::value_guard::{absorb, ValueGuard};
use crate::learning::subsumption::does_subsume;
use crate::representation::classifier::Classifier;
use crate::representation::population::Population;
use crate::support::config::Configuration;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MatureSubsumptionConfiguration {
    pub theta_exp: u32,
    pub theta_r: f64,
}

impl MatureSubsumptionConfiguration {
    pub fn new(theta_exp: u32, theta_r: f64) -> Self {
        Self { theta_exp, theta_r }
    }

    pub fn from_learning(config: &Configuration) -> Self {
        Self::new(config.theta_exp, config.theta_r)
    }
}

pub fn is_mature_subsumed<const N: usize>(
    candidate: &Classifier<N>,
    subsumer: &Classifier<N>,
    config: &MatureSubsumptionConfiguration,
) -> bool {
    does_subsume(subsumer, candidate, config.theta_exp, config.theta_r)
        && subsumer.condition.includes(&candidate.condition)
}

pub fn mature_subsumption<const N: usize>(
    rules: &[Classifier<N>],
    config: &MatureSubsumptionConfiguration,
    guard: Option<ValueGuard>,
) -> ReductionOutcome {
    absorb(
        rules,
        |candidate, survivor| is_mature_subsumed(candidate, survivor, config),
        guard,
    )
}

pub fn mature_subsumption_population<const N: usize>(
    population: &Population<N>,
    config: &MatureSubsumptionConfiguration,
    guard: Option<ValueGuard>,
) -> (Population<N>, ReductionOutcome) {
    let outcome = mature_subsumption(population.classifiers(), config, guard);
    (outcome.apply(population), outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::outcome::{RemovalReason, Verdict};
    use crate::compaction::test_support::{
        all_transitions, classifier, maximum_fitness_table, perception, random_population,
        THETA_EXP, THETA_R,
    };
    use crate::support::knowledge::anticipation_fraction;
    use crate::support::rng::ChaChaRandomSource;

    fn config() -> MatureSubsumptionConfiguration {
        MatureSubsumptionConfiguration::new(THETA_EXP, THETA_R)
    }

    fn subsumed_by(by: usize) -> Verdict {
        Verdict::Removed(RemovalReason::Subsumed { by })
    }

    #[test]
    fn mature_reliable_general_rule_removes_the_specific_one() {
        let rules = vec![
            classifier(b"10#", 0, b"##1", 0.5),
            classifier(b"1##", 0, b"##1", 0.95),
        ];
        let outcome = mature_subsumption(&rules, &config(), None);
        assert_eq!(outcome.decisions, vec![subsumed_by(1), Verdict::Kept]);
        assert_eq!(outcome.work, 1);
    }

    #[test]
    fn from_learning_copies_the_acs2_thresholds() {
        let learning = Configuration::default_protocol();
        let derived = MatureSubsumptionConfiguration::from_learning(&learning);
        assert_eq!(derived.theta_exp, learning.theta_exp);
        assert_eq!(derived.theta_r, learning.theta_r);
    }

    #[test]
    fn subsumer_must_exceed_experience_threshold() {
        let mut general = classifier(b"1##", 0, b"##1", 0.95);
        let specific = classifier(b"10#", 0, b"##1", 0.95);
        general.exp = THETA_EXP;
        let outcome = mature_subsumption(&[specific.clone(), general.clone()], &config(), None);
        assert_eq!(outcome.kept_indices(), vec![0, 1]);
        general.exp = THETA_EXP + 1;
        let outcome = mature_subsumption(&[specific, general], &config(), None);
        assert_eq!(outcome.kept_indices(), vec![1]);
    }

    #[test]
    fn subsumer_must_be_strictly_reliable() {
        let specific = classifier(b"10#", 0, b"##1", 0.95);
        let at_threshold = classifier(b"1##", 0, b"##1", THETA_R);
        let outcome = mature_subsumption(&[specific, at_threshold], &config(), None);
        assert_eq!(outcome.kept_indices(), vec![0, 1]);
    }

    #[test]
    fn marked_subsumer_never_removes_anything() {
        let specific = classifier(b"10#", 0, b"##1", 0.95);
        let mut general = classifier(b"1##", 0, b"##1", 0.95);
        general.set_mark(&perception(b"110"));
        let outcome = mature_subsumption(&[specific, general], &config(), None);
        assert_eq!(outcome.kept_indices(), vec![0, 1]);
    }

    #[test]
    fn marked_candidate_can_still_be_removed() {
        let mut specific = classifier(b"10#", 0, b"##1", 0.95);
        specific.set_mark(&perception(b"110"));
        let general = classifier(b"1##", 0, b"##1", 0.95);
        let outcome = mature_subsumption(&[specific, general], &config(), None);
        assert_eq!(outcome.kept_indices(), vec![1]);
    }

    #[test]
    fn identical_rules_are_not_removed_by_subsumption() {
        let rules = vec![
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
        ];
        assert_eq!(mature_subsumption(&rules, &config(), None).kept_indices(), vec![0, 1]);
    }

    #[test]
    fn different_action_or_effect_keeps_both() {
        let action = vec![
            classifier(b"10#", 0, b"##1", 0.95),
            classifier(b"1##", 1, b"##1", 0.95),
        ];
        assert_eq!(mature_subsumption(&action, &config(), None).kept_indices(), vec![0, 1]);
        let effect = vec![
            classifier(b"10#", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"###", 0.95),
        ];
        assert_eq!(mature_subsumption(&effect, &config(), None).kept_indices(), vec![0, 1]);
    }

    #[test]
    fn compatible_but_not_included_condition_is_kept_although_acs2_would_reject_it() {
        let rules = vec![
            classifier(b"#00", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
        ];
        assert!(does_subsume(&rules[1], &rules[0], THETA_EXP, THETA_R));
        let outcome = mature_subsumption(&rules, &config(), None);
        assert_eq!(outcome.kept_indices(), vec![0, 1]);
    }

    #[test]
    fn value_guard_blocks_a_weaker_mature_subsumer() {
        let mut specific = classifier(b"10#", 0, b"##1", 0.95);
        let mut general = classifier(b"1##", 0, b"##1", 0.95);
        specific.r = 0.9;
        general.r = 0.5;
        let rules = vec![specific, general];
        assert_eq!(
            mature_subsumption(&rules, &config(), Some(ValueGuard::exact())).kept_indices(),
            vec![0, 1]
        );
        assert_eq!(mature_subsumption(&rules, &config(), None).kept_indices(), vec![1]);
    }

    #[test]
    fn population_helper_leaves_the_input_untouched() {
        let population = Population::from_classifiers(vec![
            classifier(b"10#", 0, b"##1", 0.5),
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"0##", 1, b"###", 0.4),
        ]);
        let (reduced, outcome) = mature_subsumption_population(&population, &config(), None);
        assert_eq!(population.len(), 3);
        assert_eq!(reduced.len(), 2);
        assert_eq!(outcome.removed_indices(), vec![0]);
    }

    fn knowledge(population: &Population<3>) -> f64 {
        anticipation_fraction(population, THETA_R, all_transitions())
    }

    #[test]
    fn random_populations_keep_knowledge_and_every_removed_rule_would_be_rejected() {
        let mut rng = ChaChaRandomSource::from_seed(21);
        let mut removed_total = 0;
        for _ in 0..300 {
            let population = random_population(&mut rng, 30);
            let (reduced, outcome) = mature_subsumption_population(&population, &config(), None);
            removed_total += outcome.removed_count();
            assert_eq!(knowledge(&population), knowledge(&reduced));
            for (index, verdict) in outcome.decisions.iter().enumerate() {
                if let Verdict::Removed(RemovalReason::Subsumed { by }) = verdict {
                    assert_eq!(outcome.decisions[*by], Verdict::Kept);
                    assert!(does_subsume(
                        population.get(*by),
                        population.get(index),
                        THETA_EXP,
                        THETA_R
                    ));
                }
            }
            let (_, second) = mature_subsumption_population(&reduced, &config(), None);
            assert_eq!(second.removed_count(), 0);
            let (repeat, repeat_outcome) = mature_subsumption_population(&population, &config(), None);
            assert_eq!(repeat_outcome, outcome);
            assert_eq!(repeat.classifiers(), reduced.classifiers());
        }
        assert!(removed_total > 0);
    }

    #[test]
    fn exact_guard_keeps_knowledge_and_every_value_on_random_populations() {
        let mut rng = ChaChaRandomSource::from_seed(22);
        let mut removed_total = 0;
        for _ in 0..300 {
            let population = random_population(&mut rng, 30);
            let (reduced, outcome) =
                mature_subsumption_population(&population, &config(), Some(ValueGuard::exact()));
            removed_total += outcome.removed_count();
            assert_eq!(knowledge(&population), knowledge(&reduced));
            assert_eq!(maximum_fitness_table(&population), maximum_fitness_table(&reduced));
            let (_, second) =
                mature_subsumption_population(&reduced, &config(), Some(ValueGuard::exact()));
            assert_eq!(second.removed_count(), 0);
            let (_, repeat) =
                mature_subsumption_population(&population, &config(), Some(ValueGuard::exact()));
            assert_eq!(repeat, outcome);
        }
        assert!(removed_total > 0);
    }

    #[test]
    fn tolerance_bounds_the_value_change_on_random_populations() {
        let mut rng = ChaChaRandomSource::from_seed(23);
        let tolerance = 0.15;
        for _ in 0..200 {
            let population = random_population(&mut rng, 30);
            let (reduced, _) = mature_subsumption_population(
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
        }
    }
}
