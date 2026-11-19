use crate::representation::classifier::Classifier;
use crate::representation::symbol::Symbol;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RuleKey<const N: usize> {
    pub condition: [Symbol; N],
    pub action: Option<usize>,
    pub effect: [Symbol; N],
}

impl<const N: usize> RuleKey<N> {
    pub fn of(classifier: &Classifier<N>) -> Self {
        Self {
            condition: classifier.condition.symbols,
            action: classifier.action,
            effect: classifier.effect.symbols,
        }
    }
}
