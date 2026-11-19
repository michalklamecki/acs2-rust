use std::collections::{BTreeMap, BTreeSet};

use crate::compaction::outcome::{ReductionOutcome, Verdict};
use crate::compaction::rule_key::RuleKey;
use crate::representation::classifier::Classifier;
use crate::representation::population::Population;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Restoration {
    pub removed: usize,
    pub restored: usize,
}

impl Restoration {
    pub fn fraction(&self) -> f64 {
        if self.removed == 0 {
            return 0.0;
        }
        self.restored as f64 / self.removed as f64
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RemovalLedger<const N: usize> {
    removed_at: BTreeMap<RuleKey<N>, u64>,
}

impl<const N: usize> Default for RemovalLedger<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> RemovalLedger<N> {
    pub fn new() -> Self {
        Self {
            removed_at: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.removed_at.len()
    }

    pub fn is_empty(&self) -> bool {
        self.removed_at.is_empty()
    }

    pub fn removed_at(&self, classifier: &Classifier<N>) -> Option<u64> {
        self.removed_at.get(&RuleKey::of(classifier)).copied()
    }

    pub fn record<'a, I>(&mut self, interaction: u64, removed: I)
    where
        I: IntoIterator<Item = &'a Classifier<N>>,
    {
        for classifier in removed {
            self.removed_at.insert(RuleKey::of(classifier), interaction);
        }
    }

    pub fn record_outcome(
        &mut self,
        interaction: u64,
        before: &Population<N>,
        outcome: &ReductionOutcome,
    ) {
        assert_eq!(before.len(), outcome.decisions.len());
        let removed = outcome
            .decisions
            .iter()
            .enumerate()
            .filter(|(_, verdict)| **verdict != Verdict::Kept)
            .map(|(index, _)| before.get(index));
        self.record(interaction, removed);
    }

    pub fn restoration(&self, population: &Population<N>) -> Restoration {
        self.restoration_since(population, 0)
    }

    pub fn restoration_since(&self, population: &Population<N>, interaction: u64) -> Restoration {
        let present: BTreeSet<RuleKey<N>> =
            population.iter().map(RuleKey::of).collect();
        let mut restoration = Restoration {
            removed: 0,
            restored: 0,
        };
        for (key, at) in &self.removed_at {
            if *at < interaction {
                continue;
            }
            restoration.removed += 1;
            restoration.restored += usize::from(present.contains(key));
        }
        restoration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::outcome::RemovalReason;
    use crate::compaction::test_support::classifier;

    fn rules() -> Vec<Classifier<3>> {
        vec![
            classifier(b"10#", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"0##", 1, b"###", 0.4),
        ]
    }

    fn removal_of(indices: &[usize]) -> ReductionOutcome {
        let mut outcome = ReductionOutcome::keeping_all(3, 0);
        for &index in indices {
            outcome.decisions[index] = Verdict::Removed(RemovalReason::Evicted);
        }
        outcome
    }

    #[test]
    fn restored_fraction_counts_removed_rules_that_reappear() {
        let before = Population::from_classifiers(rules());
        let mut ledger = RemovalLedger::new();
        ledger.record_outcome(100, &before, &removal_of(&[0, 2]));
        assert_eq!(ledger.len(), 2);

        let kept = removal_of(&[0, 2]).apply(&before);
        let none_back = ledger.restoration(&kept);
        assert_eq!(none_back, Restoration { removed: 2, restored: 0 });
        assert_eq!(none_back.fraction(), 0.0);

        let mut later = kept.classifiers().to_vec();
        let mut reborn = classifier(b"10#", 0, b"##1", 0.1);
        reborn.exp = 0;
        later.push(reborn);
        let one_back = ledger.restoration(&Population::from_classifiers(later));
        assert_eq!(one_back, Restoration { removed: 2, restored: 1 });
        assert_eq!(one_back.fraction(), 0.5);
    }

    #[test]
    fn empty_ledger_reports_zero() {
        let ledger = RemovalLedger::<3>::new();
        assert!(ledger.is_empty());
        let report = ledger.restoration(&Population::from_classifiers(rules()));
        assert_eq!(report, Restoration { removed: 0, restored: 0 });
        assert_eq!(report.fraction(), 0.0);
    }

    #[test]
    fn restoration_since_counts_only_recent_removals() {
        let before = Population::from_classifiers(rules());
        let mut ledger = RemovalLedger::new();
        ledger.record_outcome(100, &before, &removal_of(&[0]));
        ledger.record_outcome(500, &before, &removal_of(&[2]));
        let report = ledger.restoration_since(&before, 300);
        assert_eq!(report, Restoration { removed: 1, restored: 1 });
        assert_eq!(ledger.removed_at(before.get(0)), Some(100));
        assert_eq!(ledger.removed_at(before.get(1)), None);
    }

    #[test]
    fn removing_again_updates_the_interaction() {
        let before = Population::from_classifiers(rules());
        let mut ledger = RemovalLedger::new();
        ledger.record_outcome(100, &before, &removal_of(&[0]));
        ledger.record_outcome(900, &before, &removal_of(&[0]));
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.removed_at(before.get(0)), Some(900));
    }

    #[test]
    fn record_accepts_any_iterator_of_rules() {
        let mut ledger = RemovalLedger::new();
        ledger.record(7, rules().iter());
        assert_eq!(ledger.len(), 3);
    }
}
