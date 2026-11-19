use crate::compaction::evidence::{is_reliably_anticipated, states_of, StateMatches};
use crate::learning::rl::{BootstrapEstimator, MaxFitnessBootstrap};
use crate::representation::population::Population;
use crate::support::knowledge::Transition;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AuditThresholds {
    pub max_coverage_loss: f64,
    pub max_no_change_coverage_loss: f64,
    pub max_value_change: f64,
    pub max_greedy_change_fraction: f64,
}

impl AuditThresholds {
    pub fn new(
        max_coverage_loss: f64,
        max_no_change_coverage_loss: f64,
        max_value_change: f64,
        max_greedy_change_fraction: f64,
    ) -> Self {
        Self {
            max_coverage_loss,
            max_no_change_coverage_loss,
            max_value_change,
            max_greedy_change_fraction,
        }
    }

    pub fn lossless() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Violation {
    CoverageLoss,
    NoChangeCoverageLoss,
    ValueChange,
    GreedyChange,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AuditVerdict {
    pub violations: Vec<Violation>,
}

impl AuditVerdict {
    pub fn accepted(&self) -> bool {
        self.violations.is_empty()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CoverageCount {
    pub transitions: usize,
    pub covered: usize,
}

impl CoverageCount {
    pub fn fraction(&self) -> f64 {
        if self.transitions == 0 {
            return 0.0;
        }
        self.covered as f64 / self.transitions as f64
    }

    fn loss_against(&self, after: &CoverageCount) -> f64 {
        if self.transitions == 0 {
            return 0.0;
        }
        (self.covered as f64 - after.covered as f64) / self.transitions as f64
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Coverage {
    pub all: CoverageCount,
    pub state_changing: CoverageCount,
    pub no_change: CoverageCount,
}

#[derive(Clone, PartialEq, Debug)]
pub struct AuditReport {
    pub coverage_before: Coverage,
    pub coverage_after: Coverage,
    pub states: usize,
    pub value_pairs: usize,
    pub max_value_change: f64,
    pub mean_value_change: f64,
    pub greedy_changes: usize,
    pub work: u64,
}

impl AuditReport {
    pub fn coverage_loss(&self) -> f64 {
        self.coverage_before.all.loss_against(&self.coverage_after.all)
    }

    pub fn state_changing_coverage_loss(&self) -> f64 {
        self.coverage_before
            .state_changing
            .loss_against(&self.coverage_after.state_changing)
    }

    pub fn no_change_coverage_loss(&self) -> f64 {
        self.coverage_before
            .no_change
            .loss_against(&self.coverage_after.no_change)
    }

    pub fn greedy_change_fraction(&self) -> f64 {
        if self.states == 0 {
            return 0.0;
        }
        self.greedy_changes as f64 / self.states as f64
    }

    pub fn verdict(&self, thresholds: &AuditThresholds) -> AuditVerdict {
        let checks = [
            (
                self.coverage_loss() > thresholds.max_coverage_loss,
                Violation::CoverageLoss,
            ),
            (
                self.no_change_coverage_loss() > thresholds.max_no_change_coverage_loss,
                Violation::NoChangeCoverageLoss,
            ),
            (
                self.max_value_change > thresholds.max_value_change,
                Violation::ValueChange,
            ),
            (
                self.greedy_change_fraction() > thresholds.max_greedy_change_fraction,
                Violation::GreedyChange,
            ),
        ];
        AuditVerdict {
            violations: checks
                .into_iter()
                .filter(|(violated, _)| *violated)
                .map(|(_, violation)| violation)
                .collect(),
        }
    }
}

fn coverage_of<const N: usize>(
    population: &Population<N>,
    matches: &StateMatches<N>,
    transitions: &[Transition<N>],
    theta_r: f64,
) -> Coverage {
    let mut coverage = Coverage::default();
    for transition in transitions {
        let action_set = matches.action_set(population, &transition.p0, transition.action);
        let covered = is_reliably_anticipated(population, &action_set, transition, theta_r);
        let class = if transition.p0 == transition.p1 {
            &mut coverage.no_change
        } else {
            &mut coverage.state_changing
        };
        for count in [class, &mut coverage.all] {
            count.transitions += 1;
            count.covered += usize::from(covered);
        }
    }
    coverage
}

fn action_values<const N: usize>(
    population: &Population<N>,
    matches: &StateMatches<N>,
    actions: usize,
) -> Vec<Vec<f64>> {
    matches
        .states()
        .iter()
        .map(|state| {
            (0..actions)
                .map(|action| {
                    let action_set = matches.action_set(population, state, action);
                    MaxFitnessBootstrap.estimate(population, &action_set)
                })
                .collect()
        })
        .collect()
}

fn greedy_actions(values: &[f64]) -> Vec<usize> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| **value == maximum)
        .map(|(action, _)| action)
        .collect()
}

pub fn audit_reduction<const N: usize>(
    before: &Population<N>,
    after: &Population<N>,
    transitions: &[Transition<N>],
    actions: usize,
    theta_r: f64,
) -> AuditReport {
    let matches_before = StateMatches::collect(before, states_of(transitions));
    let matches_after = StateMatches::collect(after, states_of(transitions));
    let values_before = action_values(before, &matches_before, actions);
    let values_after = action_values(after, &matches_after, actions);

    let mut max_change: f64 = 0.0;
    let mut total_change = 0.0;
    let mut pairs = 0usize;
    let mut greedy_changes = 0usize;
    for (row_before, row_after) in values_before.iter().zip(&values_after) {
        for (value_before, value_after) in row_before.iter().zip(row_after) {
            let change = (value_after - value_before).abs();
            max_change = max_change.max(change);
            total_change += change;
            pairs += 1;
        }
        if greedy_actions(row_before) != greedy_actions(row_after) {
            greedy_changes += 1;
        }
    }

    AuditReport {
        coverage_before: coverage_of(before, &matches_before, transitions, theta_r),
        coverage_after: coverage_of(after, &matches_after, transitions, theta_r),
        states: matches_before.states().len(),
        value_pairs: pairs,
        max_value_change: max_change,
        mean_value_change: if pairs == 0 {
            0.0
        } else {
            total_change / pairs as f64
        },
        greedy_changes,
        work: matches_before.work() + matches_after.work(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::mature_subsumption::{
        mature_subsumption_population, MatureSubsumptionConfiguration,
    };
    use crate::compaction::test_support::{
        all_transitions, classifier, environment_transitions, grounded_population, transition,
        ACTIONS, THETA_EXP, THETA_R,
    };
    use crate::compaction::value_guard::ValueGuard;
    use crate::representation::classifier::Classifier;
    use crate::support::knowledge::anticipation_fraction;
    use crate::support::rng::ChaChaRandomSource;

    fn sample() -> Vec<Transition<3>> {
        vec![
            transition(b"100", 0, b"101"),
            transition(b"000", 0, b"000"),
            transition(b"110", 1, b"100"),
        ]
    }

    fn rules() -> Vec<Classifier<3>> {
        vec![
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"0##", 0, b"###", 0.95),
            classifier(b"#1#", 1, b"#0#", 0.95),
        ]
    }

    fn without(index: usize) -> Population<3> {
        let mut kept = rules();
        kept.remove(index);
        Population::from_classifiers(kept)
    }

    fn run(before: &Population<3>, after: &Population<3>) -> AuditReport {
        audit_reduction(before, after, &sample(), ACTIONS, THETA_R)
    }

    #[test]
    fn identical_populations_are_accepted_under_lossless_thresholds() {
        let population = Population::from_classifiers(rules());
        let report = run(&population, &population);
        assert_eq!(report.coverage_before, report.coverage_after);
        assert_eq!(report.coverage_before.all, CoverageCount { transitions: 3, covered: 3 });
        assert_eq!(report.coverage_loss(), 0.0);
        assert_eq!(report.max_value_change, 0.0);
        assert_eq!(report.mean_value_change, 0.0);
        assert_eq!(report.greedy_changes, 0);
        assert!(report.verdict(&AuditThresholds::lossless()).accepted());
    }

    #[test]
    fn coverage_matches_the_knowledge_measure() {
        let population = Population::from_classifiers(rules());
        let report = run(&population, &without(0));
        let copy = |transitions: Vec<Transition<3>>| {
            transitions
                .iter()
                .map(|t| Transition::new(t.p0, t.action, t.p1))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            report.coverage_before.all.fraction(),
            anticipation_fraction(&population, THETA_R, copy(sample()))
        );
        assert_eq!(
            report.coverage_after.all.fraction(),
            anticipation_fraction(&without(0), THETA_R, copy(sample()))
        );
    }

    #[test]
    fn losing_a_state_changing_anticipator_is_rejected() {
        let population = Population::from_classifiers(rules());
        let report = run(&population, &without(0));
        assert_eq!(report.coverage_after.state_changing.covered, 1);
        assert_eq!(report.coverage_before.state_changing.transitions, 2);
        assert!((report.coverage_loss() - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(report.no_change_coverage_loss(), 0.0);
        let verdict = report.verdict(&AuditThresholds::new(0.0, 0.0, 10.0, 1.0));
        assert_eq!(verdict.violations, vec![Violation::CoverageLoss]);
        let relaxed = report.verdict(&AuditThresholds::new(0.5, 0.0, 10.0, 1.0));
        assert!(relaxed.accepted());
    }

    #[test]
    fn wall_bump_coverage_is_reported_and_judged_separately() {
        let population = Population::from_classifiers(rules());
        let report = run(&population, &without(1));
        assert_eq!(report.coverage_before.no_change, CoverageCount { transitions: 1, covered: 1 });
        assert_eq!(report.coverage_after.no_change, CoverageCount { transitions: 1, covered: 0 });
        assert_eq!(report.no_change_coverage_loss(), 1.0);
        assert_eq!(report.state_changing_coverage_loss(), 0.0);
        let verdict = report.verdict(&AuditThresholds::new(0.5, 0.0, 10.0, 1.0));
        assert_eq!(verdict.violations, vec![Violation::NoChangeCoverageLoss]);
    }

    #[test]
    fn value_change_reports_maximum_and_mean_over_sample_states_and_actions() {
        let before = Population::from_classifiers(vec![classifier(b"1##", 0, b"##1", 0.9)]);
        let mut weaker = classifier(b"1##", 0, b"##1", 0.9);
        weaker.r = 0.5;
        let after = Population::from_classifiers(vec![weaker]);
        let transitions = vec![transition(b"100", 0, b"101"), transition(b"000", 0, b"000")];
        let report = audit_reduction(&before, &after, &transitions, ACTIONS, THETA_R);
        assert_eq!(report.states, 2);
        assert_eq!(report.value_pairs, 4);
        assert!((report.max_value_change - 0.45).abs() < 1e-12);
        assert!((report.mean_value_change - 0.45 / 4.0).abs() < 1e-12);
        let verdict = report.verdict(&AuditThresholds::new(1.0, 1.0, 0.1, 1.0));
        assert_eq!(verdict.violations, vec![Violation::ValueChange]);
    }

    #[test]
    fn greedy_change_is_a_change_of_the_argmax_action_set() {
        let mut stronger = classifier(b"1##", 0, b"##1", 0.9);
        stronger.r = 0.9;
        let mut other = classifier(b"1##", 1, b"##1", 0.9);
        other.r = 0.1;
        let before = Population::from_classifiers(vec![stronger.clone(), other.clone()]);
        let mut flipped = other.clone();
        flipped.r = 1.0;
        let after = Population::from_classifiers(vec![stronger.clone(), flipped]);
        let transitions = vec![transition(b"100", 0, b"101")];
        let report = audit_reduction(&before, &after, &transitions, ACTIONS, THETA_R);
        assert_eq!(report.greedy_changes, 1);
        assert_eq!(report.greedy_change_fraction(), 1.0);
        assert_eq!(
            report.verdict(&AuditThresholds::new(1.0, 1.0, 10.0, 0.5)).violations,
            vec![Violation::GreedyChange]
        );

        let mut tie_other = other.clone();
        tie_other.r = 0.9;
        let tie = Population::from_classifiers(vec![stronger, tie_other]);
        let tie_report = audit_reduction(&before, &tie, &transitions, ACTIONS, THETA_R);
        assert_eq!(tie_report.greedy_changes, 1);
    }

    #[test]
    fn unchanged_values_with_ties_do_not_count_as_greedy_change() {
        let empty = Population::<3>::new();
        let report = audit_reduction(&empty, &empty, &sample(), ACTIONS, THETA_R);
        assert_eq!(report.greedy_changes, 0);
        assert_eq!(report.coverage_before.all.covered, 0);
    }

    #[test]
    fn work_counts_match_tests_of_both_populations_on_distinct_states() {
        let population = Population::from_classifiers(rules());
        let mut transitions = sample();
        transitions.push(transition(b"100", 1, b"100"));
        let report = audit_reduction(&population, &without(0), &transitions, ACTIONS, THETA_R);
        assert_eq!(report.work, 3 * 3 + 3 * 2);
    }

    #[test]
    fn empty_sample_reports_nothing_and_accepts() {
        let population = Population::from_classifiers(rules());
        let report = audit_reduction(&population, &Population::new(), &[], ACTIONS, THETA_R);
        assert_eq!(report.states, 0);
        assert_eq!(report.coverage_loss(), 0.0);
        assert_eq!(report.greedy_change_fraction(), 0.0);
        assert!(report.verdict(&AuditThresholds::lossless()).accepted());
        assert_eq!(report.work, 0);
    }

    #[test]
    fn exact_mature_subsumption_always_passes_a_lossless_audit() {
        let mut rng = ChaChaRandomSource::from_seed(61);
        let config = MatureSubsumptionConfiguration::new(THETA_EXP, THETA_R);
        for _ in 0..100 {
            let population = grounded_population(&mut rng, 30);
            let (reduced, _) =
                mature_subsumption_population(&population, &config, Some(ValueGuard::exact()));
            for transitions in [all_transitions(), environment_transitions()] {
                let report = audit_reduction(&population, &reduced, &transitions, ACTIONS, THETA_R);
                assert!(report.verdict(&AuditThresholds::lossless()).accepted());
            }
        }
    }

    #[test]
    fn tolerance_stays_within_the_audited_value_bound() {
        let mut rng = ChaChaRandomSource::from_seed(62);
        let config = MatureSubsumptionConfiguration::new(THETA_EXP, THETA_R);
        for _ in 0..100 {
            let population = grounded_population(&mut rng, 30);
            let (reduced, _) =
                mature_subsumption_population(&population, &config, Some(ValueGuard::new(0.1)));
            let report = audit_reduction(&population, &reduced, &all_transitions(), ACTIONS, THETA_R);
            assert!(report.max_value_change <= 0.1);
            assert!(report.coverage_loss() <= 0.0);
        }
    }
}
