use std::collections::{BTreeMap, BTreeSet};

use crate::learning::rl::{BootstrapEstimator, MaxFitnessBootstrap};
use crate::representation::perception::Perception;
use crate::representation::population::{ClassifierRef, Population};
use crate::representation::symbol::Symbol;
use crate::support::knowledge::Transition;

pub struct StateMatches<const N: usize> {
    index: BTreeMap<[Symbol; N], usize>,
    states: Vec<Perception<N>>,
    matched: Vec<Vec<ClassifierRef>>,
    work: u64,
}

impl<const N: usize> StateMatches<N> {
    pub fn collect<'a, I>(population: &Population<N>, perceptions: I) -> Self
    where
        I: IntoIterator<Item = &'a Perception<N>>,
    {
        let mut collected = Self {
            index: BTreeMap::new(),
            states: Vec::new(),
            matched: Vec::new(),
            work: 0,
        };
        for perception in perceptions {
            if collected.index.contains_key(&perception.symbols) {
                continue;
            }
            let matched: Vec<ClassifierRef> = (0..population.len())
                .filter(|&reference| population.get(reference).does_match(perception))
                .collect();
            collected.work += population.len() as u64;
            collected
                .index
                .insert(perception.symbols, collected.states.len());
            collected.states.push(*perception);
            collected.matched.push(matched);
        }
        collected
    }

    pub fn work(&self) -> u64 {
        self.work
    }

    pub fn states(&self) -> &[Perception<N>] {
        &self.states
    }

    pub fn matched(&self, state: &Perception<N>) -> &[ClassifierRef] {
        let position = self
            .index
            .get(&state.symbols)
            .expect("state was not part of the collected sample");
        &self.matched[*position]
    }

    pub fn action_set(
        &self,
        population: &Population<N>,
        state: &Perception<N>,
        action: usize,
    ) -> Vec<ClassifierRef> {
        population.form_action_set(self.matched(state), action)
    }
}

pub fn states_of<const N: usize>(transitions: &[Transition<N>]) -> impl Iterator<Item = &Perception<N>> {
    transitions.iter().map(|transition| &transition.p0)
}

pub fn is_reliably_anticipated<const N: usize>(
    population: &Population<N>,
    action_set: &[ClassifierRef],
    transition: &Transition<N>,
    theta_r: f64,
) -> bool {
    best_reliable_anticipator(population, action_set, transition, theta_r).is_some()
}

pub fn best_reliable_anticipator<const N: usize>(
    population: &Population<N>,
    action_set: &[ClassifierRef],
    transition: &Transition<N>,
    theta_r: f64,
) -> Option<ClassifierRef> {
    let mut best: Option<ClassifierRef> = None;
    for &reference in action_set {
        let classifier = population.get(reference);
        if !classifier.is_reliable(theta_r)
            || !classifier.does_anticipate_correctly(&transition.p0, &transition.p1)
        {
            continue;
        }
        let better = match best {
            Some(current) => classifier.q > population.get(current).q,
            None => true,
        };
        if better {
            best = Some(reference);
        }
    }
    best
}

pub fn action_set_carriers<const N: usize>(
    population: &Population<N>,
    action_set: &[ClassifierRef],
) -> Vec<ClassifierRef> {
    let maximum = MaxFitnessBootstrap.estimate(population, action_set);
    if maximum <= 0.0 {
        return Vec::new();
    }
    action_set
        .iter()
        .copied()
        .filter(|&reference| {
            let classifier = population.get(reference);
            classifier.does_anticipate_change() && classifier.fitness() == maximum
        })
        .collect()
}

pub fn sample_value_carriers<const N: usize>(
    population: &Population<N>,
    matches: &StateMatches<N>,
    transitions: &[Transition<N>],
) -> BTreeSet<ClassifierRef> {
    let mut seen: BTreeSet<([Symbol; N], usize)> = BTreeSet::new();
    let mut carriers = BTreeSet::new();
    for transition in transitions {
        if !seen.insert((transition.p0.symbols, transition.action)) {
            continue;
        }
        let action_set = matches.action_set(population, &transition.p0, transition.action);
        carriers.extend(action_set_carriers(population, &action_set));
    }
    carriers
}
