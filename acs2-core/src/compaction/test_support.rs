use crate::learning::rl::{BootstrapEstimator, MaxFitnessBootstrap};
use crate::representation::classifier::Classifier;
use crate::representation::perception::Perception;
use crate::representation::population::Population;
use crate::representation::symbol::Symbol;
use crate::support::config::Configuration;
use crate::support::knowledge::Transition;
use crate::support::rng::RandomSource;

pub const THETA_R: f64 = 0.9;
pub const THETA_EXP: u32 = 20;
pub const ACTIONS: usize = 2;

pub fn symbol(byte: u8) -> Symbol {
    if byte == b'#' {
        Symbol::Wildcard
    } else {
        Symbol::Token(byte)
    }
}

pub fn classifier(condition: &[u8; 3], action: usize, effect: &[u8; 3], q: f64) -> Classifier<3> {
    let mut classifier = Classifier::general(Some(action), &Configuration::default_protocol());
    for index in 0..3 {
        classifier.condition.set(index, symbol(condition[index]));
        classifier.effect.set(index, symbol(effect[index]));
    }
    classifier.q = q;
    classifier.r = 1.0;
    classifier.exp = THETA_EXP + 10;
    classifier
}

pub fn perception(text: &[u8; 3]) -> Perception<3> {
    Perception::new(text.map(symbol))
}

pub fn transition(before: &[u8; 3], action: usize, after: &[u8; 3]) -> Transition<3> {
    Transition::new(perception(before), action, perception(after))
}

fn random_symbol(rng: &mut dyn RandomSource, wildcard_probability: f64) -> Symbol {
    if rng.gen_bool(wildcard_probability) {
        Symbol::Wildcard
    } else {
        Symbol::Token(b'0' + rng.gen_range(2) as u8)
    }
}

fn random_perception(rng: &mut dyn RandomSource) -> Perception<3> {
    Perception::new(core::array::from_fn(|_| Symbol::Token(b'0' + rng.gen_range(2) as u8)))
}

pub fn random_classifier(rng: &mut dyn RandomSource) -> Classifier<3> {
    let mut classifier =
        Classifier::general(Some(rng.gen_range(ACTIONS)), &Configuration::default_protocol());
    for index in 0..3 {
        classifier.condition.set(index, random_symbol(rng, 0.5));
        classifier.effect.set(index, random_symbol(rng, 0.7));
    }
    classifier.q = if rng.gen_bool(0.5) {
        0.9 + 0.1 * rng.gen_unit()
    } else {
        rng.gen_unit()
    };
    classifier.r = if rng.gen_bool(0.2) { 0.0 } else { rng.gen_unit() };
    classifier.exp = rng.gen_range(50) as u32;
    if rng.gen_bool(0.3) {
        let marked = random_perception(rng);
        classifier.set_mark(&marked);
    }
    classifier
}

pub fn random_population(rng: &mut dyn RandomSource, max_size: usize) -> Population<3> {
    let size = 1 + rng.gen_range(max_size);
    Population::from_classifiers((0..size).map(|_| random_classifier(rng)).collect())
}

pub fn all_perceptions() -> Vec<Perception<3>> {
    (0..8u8)
        .map(|bits| {
            Perception::new(core::array::from_fn(|index| {
                Symbol::Token(b'0' + ((bits >> index) & 1))
            }))
        })
        .collect()
}

pub fn all_transitions() -> Vec<Transition<3>> {
    let perceptions = all_perceptions();
    let mut transitions = Vec::new();
    for p0 in &perceptions {
        for action in 0..ACTIONS {
            for p1 in &perceptions {
                transitions.push(Transition::new(*p0, action, *p1));
            }
        }
    }
    transitions
}

pub fn maximum_fitness_table(population: &Population<3>) -> Vec<f64> {
    let mut table = Vec::new();
    for state in all_perceptions() {
        let matched: Vec<usize> = (0..population.len())
            .filter(|&reference| population.get(reference).does_match(&state))
            .collect();
        for action in 0..ACTIONS {
            let action_set = population.form_action_set(&matched, action);
            table.push(MaxFitnessBootstrap.estimate(population, &action_set));
        }
    }
    table
}

pub fn environment_transitions() -> Vec<Transition<3>> {
    let mut transitions = Vec::new();
    for p0 in all_perceptions() {
        for action in 0..ACTIONS {
            let mut p1 = p0;
            let flipped = if p0.get(action) == Symbol::Token(b'0') { b'1' } else { b'0' };
            p1.symbols[action] = Symbol::Token(flipped);
            transitions.push(Transition::new(p0, action, p1));
        }
    }
    transitions
}

pub fn grounded_classifier(rng: &mut dyn RandomSource) -> Classifier<3> {
    let environment = environment_transitions();
    let chosen = &environment[rng.gen_range(environment.len())];
    let action = chosen.action;
    let mut classifier = Classifier::general(Some(action), &Configuration::default_protocol());
    for index in 0..3 {
        if index == action || rng.gen_bool(0.4) {
            classifier.condition.set(index, chosen.p0.get(index));
        }
    }
    classifier.effect.set(action, chosen.p1.get(action));
    classifier.q = if rng.gen_bool(0.7) {
        0.9 + 0.1 * rng.gen_unit()
    } else {
        rng.gen_unit()
    };
    classifier.r = if rng.gen_bool(0.2) { 0.0 } else { rng.gen_unit() };
    classifier.exp = rng.gen_range(50) as u32;
    classifier
}

pub fn grounded_population(rng: &mut dyn RandomSource, max_size: usize) -> Population<3> {
    let size = 1 + rng.gen_range(max_size);
    Population::from_classifiers(
        (0..size)
            .map(|_| {
                if rng.gen_bool(0.8) {
                    grounded_classifier(rng)
                } else {
                    random_classifier(rng)
                }
            })
            .collect(),
    )
}
