use acs2_core::compaction::trigger::Periodic;
use acs2_core::compaction::{
    fixed_strength_policy, AuditPolicy, AuditThresholds, CompactionMode, CompactionParameters,
    CompactionStrength, Operator,
};
use acs2_core::ports::environment::{Environment, StepOutcome};
use acs2_core::ports::goal::{ExactMatch, Goal, GoalOutcome, GoalStart, GoalStep};
use acs2_core::representation::classifier::Classifier;
use acs2_core::representation::perception::Perception;
use acs2_core::representation::population::Population;
use acs2_core::representation::symbol::Symbol;
use acs2_core::support::config::Configuration;
use acs2_core::trial::TruncationMode;
use acs2_trajectory::GoalEvaluator;

use super::{EpisodicConfiguration, ReplayEnvironment};

pub fn learning_configuration(do_ga: bool) -> Configuration {
    Configuration {
        number_of_possible_actions: 2,
        epsilon: 0.5,
        do_ga,
        ..Configuration::default_protocol()
    }
}

pub fn episodic_configuration() -> EpisodicConfiguration {
    EpisodicConfiguration {
        capacity: 100,
        warmup: 1,
        updates_per_step: 3,
        truncation: TruncationMode::Bootstrap,
    }
}

pub fn redundant_population(config: &Configuration) -> Population<2> {
    let rule = |condition: [Symbol; 2], value: f64| {
        let mut classifier = Classifier::general(Some(0), config);
        classifier.condition.symbols = condition;
        classifier.q = 1.0;
        classifier.r = value;
        classifier.exp = 30;
        classifier
    };
    Population::from_classifiers(vec![
        rule([Symbol::Token(0), Symbol::Token(9)], 50.0),
        rule([Symbol::Wildcard, Symbol::Token(9)], 100.0),
    ])
}

pub fn subsuming_parameters(
    mode: CompactionMode,
    every: u64,
    enforced_audit: bool,
) -> CompactionParameters {
    let strength = CompactionStrength {
        value_tolerance: 0.0,
        maturity_experience: 20,
        idle_audits: 1,
    };
    let policy = fixed_strength_policy(Periodic::new(every).unwrap(), strength).unwrap();
    let builder = CompactionParameters::builder(0.3, policy)
        .mode(mode)
        .operator(Operator::MatureSubsumption { guarded: true });
    if enforced_audit {
        builder
            .audit(AuditPolicy::Enforce(AuditThresholds::lossless()))
            .evidence_sample(8)
            .build()
            .unwrap()
    } else {
        builder.build().unwrap()
    }
}

pub fn perception(values: [u8; 2]) -> Perception<2> {
    Perception {
        symbols: values.map(Symbol::Token),
    }
}

pub struct CycleEnvironment {
    state: u8,
    steps: usize,
    length: usize,
    terminal: bool,
}

impl CycleEnvironment {
    pub fn truncating() -> Self {
        Self {
            state: 0,
            steps: 0,
            length: 3,
            terminal: false,
        }
    }

    pub fn terminating() -> Self {
        Self {
            terminal: true,
            ..Self::truncating()
        }
    }

    fn finished(&self) -> bool {
        self.steps == self.length
    }
}

impl Environment<2> for CycleEnvironment {
    fn reset(&mut self) -> Perception<2> {
        self.state = 0;
        self.steps = 0;
        perception([0, 9])
    }

    fn step(&mut self, action: usize) -> StepOutcome<2> {
        if action == 1 {
            self.state = (self.state + 1) % 3;
        }
        self.steps += 1;
        StepOutcome {
            observation: perception([self.state, 9]),
            reward: 0.0,
            terminated: self.terminal && self.finished(),
            truncated: !self.terminal && self.finished(),
            info: (),
        }
    }
}

impl GoalEvaluator<1, 1> for CycleEnvironment {
    fn outcome(&self, step: &GoalStep<1, 1>, goal: &Goal<1>) -> GoalOutcome {
        step.outcome(
            &ExactMatch {
                reward_on_reach: 1000.0,
            },
            goal,
        )
    }

    fn is_reached(&self, achieved: &Goal<1>, desired: &Goal<1>) -> bool {
        achieved == desired
    }
}

impl ReplayEnvironment<1, 1, 2> for CycleEnvironment {
    fn episode_start(&self) -> GoalStart<1, 1> {
        GoalStart {
            observation: Perception {
                symbols: [Symbol::Token(0)],
            },
            achieved: Goal::new([Symbol::Token(0)]),
            desired: Goal::new([Symbol::Token(9)]),
        }
    }

    fn last_step(&self) -> GoalStep<1, 1> {
        GoalStep {
            observation: Perception {
                symbols: [Symbol::Token(self.state)],
            },
            achieved: Goal::new([Symbol::Token(self.state)]),
            terminal_state: self.terminal && self.finished(),
            time_limit_reached: !self.terminal && self.finished(),
        }
    }
}
