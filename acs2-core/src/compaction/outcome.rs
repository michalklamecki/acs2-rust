use crate::representation::population::Population;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RemovalReason {
    Subsumed { by: usize },
    Idle { audits: u32 },
    LowQuality,
    Evicted,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Kept,
    Removed(RemovalReason),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ReductionOutcome {
    pub decisions: Vec<Verdict>,
    pub work: u64,
    pub value_vetoes: usize,
}

impl ReductionOutcome {
    pub fn keeping_all(len: usize, work: u64) -> Self {
        Self {
            decisions: vec![Verdict::Kept; len],
            work,
            value_vetoes: 0,
        }
    }

    pub fn kept_indices(&self) -> Vec<usize> {
        self.indices_where(|verdict| verdict == Verdict::Kept)
    }

    pub fn removed_indices(&self) -> Vec<usize> {
        self.indices_where(|verdict| verdict != Verdict::Kept)
    }

    pub fn removed_count(&self) -> usize {
        self.removed_indices().len()
    }

    pub fn compression_rate(&self) -> f64 {
        if self.decisions.is_empty() {
            return 0.0;
        }
        self.removed_count() as f64 / self.decisions.len() as f64
    }

    pub fn apply<const N: usize>(&self, population: &Population<N>) -> Population<N> {
        assert_eq!(self.decisions.len(), population.len());
        Population::from_classifiers(
            self.kept_indices()
                .into_iter()
                .map(|index| population.get(index).clone())
                .collect(),
        )
    }

    fn indices_where(&self, predicate: impl Fn(Verdict) -> bool) -> Vec<usize> {
        self.decisions
            .iter()
            .enumerate()
            .filter(|(_, verdict)| predicate(**verdict))
            .map(|(index, _)| index)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::test_support::classifier;

    fn outcome() -> ReductionOutcome {
        ReductionOutcome {
            decisions: vec![
                Verdict::Kept,
                Verdict::Removed(RemovalReason::Idle { audits: 2 }),
                Verdict::Removed(RemovalReason::Subsumed { by: 0 }),
                Verdict::Kept,
            ],
            work: 9,
            value_vetoes: 0,
        }
    }

    #[test]
    fn indices_and_rates_follow_the_decisions() {
        let outcome = outcome();
        assert_eq!(outcome.kept_indices(), vec![0, 3]);
        assert_eq!(outcome.removed_indices(), vec![1, 2]);
        assert_eq!(outcome.removed_count(), 2);
        assert_eq!(outcome.compression_rate(), 0.5);
        assert_eq!(ReductionOutcome::keeping_all(0, 0).compression_rate(), 0.0);
    }

    #[test]
    fn apply_builds_a_new_population_in_the_original_order() {
        let population = Population::from_classifiers(vec![
            classifier(b"1##", 0, b"##1", 0.9),
            classifier(b"0##", 0, b"##1", 0.9),
            classifier(b"01#", 0, b"##1", 0.9),
            classifier(b"11#", 0, b"##1", 0.9),
        ]);
        let reduced = outcome().apply(&population);
        assert_eq!(population.len(), 4);
        assert_eq!(reduced.len(), 2);
        assert_eq!(reduced.get(0), population.get(0));
        assert_eq!(reduced.get(1), population.get(3));
    }

    #[test]
    #[should_panic]
    fn apply_rejects_an_outcome_of_another_population() {
        let population = Population::from_classifiers(vec![classifier(b"1##", 0, b"##1", 0.9)]);
        outcome().apply(&population);
    }
}
