use std::mem::size_of;

use acs2_agents::episodic_replay::EpisodicReplay;
use acs2_core::representation::classifier::Classifier;
use acs2_core::support::knowledge::{anticipation_fraction, Transition};
use acs2_core::support::rng::RandomSource;
use serde_json::{json, Value};

pub fn replay_updates<const S: usize, const G: usize, const M: usize, R, P>(
    learner: &EpisodicReplay<S, G, M, R, P>,
) -> u64
where
    R: RandomSource,
    P: RandomSource,
{
    let counts = learner.counts();
    counts.full + counts.values_only
}

pub fn learning_update_counts<const S: usize, const G: usize, const M: usize, R, P>(
    learner: &EpisodicReplay<S, G, M, R, P>,
) -> Value
where
    R: RandomSource,
    P: RandomSource,
{
    let counts = learner.counts();
    json!({"full": counts.full, "values_only": counts.values_only,
        "skipped_warmup_steps": counts.skipped_warmup_steps})
}

pub fn population_logical_bytes<const S: usize, const G: usize, const M: usize, R, P>(
    learner: &EpisodicReplay<S, G, M, R, P>,
) -> usize
where
    R: RandomSource,
    P: RandomSource,
{
    learner.population().len() * size_of::<Classifier<M>>()
}

pub fn population_mark_entries<const S: usize, const G: usize, const M: usize, R, P>(
    learner: &EpisodicReplay<S, G, M, R, P>,
) -> usize
where
    R: RandomSource,
    P: RandomSource,
{
    learner
        .population()
        .classifiers()
        .iter()
        .map(|classifier| {
            classifier
                .mark
                .attributes
                .iter()
                .map(|attribute| attribute.len())
                .sum::<usize>()
        })
        .sum()
}

pub fn knowledge<const S: usize, const G: usize, const M: usize, R, P>(
    learner: &EpisodicReplay<S, G, M, R, P>,
    transitions: &[Transition<M>],
) -> f64
where
    R: RandomSource,
    P: RandomSource,
{
    anticipation_fraction(
        learner.population(),
        learner.config().theta_r,
        transitions
            .iter()
            .map(|transition| Transition::new(transition.p0, transition.action, transition.p1)),
    )
}

pub fn replay_diagnostics<const S: usize, const G: usize, const M: usize, R, P>(
    learner: &EpisodicReplay<S, G, M, R, P>,
    name: &str,
) -> Value
where
    R: RandomSource,
    P: RandomSource,
{
    let counters = learner.sampler().counters();
    let mut diagnostics = crate::trajectory::replay_diagnostics(counters);
    let draws = counters.pooled().draws;
    diagnostics["realized_relabeled_share"] =
        json!((draws > 0).then(|| counters.relabeled.draws as f64 / draws as f64));
    diagnostics["stored_steps"] = json!(learner.store().len());
    diagnostics["stored_episodes"] = json!(learner.store().episodes().count());
    if matches!(name, "acs2her_nontrivial" | "acs2her_shortened") {
        let walls = learner.wall_counters();
        diagnostics["wall_bumps"] = json!({
            "relabeled_draws": walls.relabeled_draws,
            "unchanged_state": walls.unchanged_state,
            "unchanged_state_terminal": walls.unchanged_state_terminal,
            "proposals_equal_draws": true,
        });
    }
    diagnostics
}

pub fn agent_parameters<const S: usize, const G: usize, const M: usize, R, P>(
    learner: &EpisodicReplay<S, G, M, R, P>,
) -> Value
where
    R: RandomSource,
    P: RandomSource,
{
    let replay = learner.replay_configuration();
    let sampler = learner.sampler().configuration();
    json!({"replay": true, "schedule": "after_complete_episode", "replay_capacity": replay.capacity,
        "replay_warmup": replay.warmup, "warmup_rule": "retained_steps_after_episode_insertion_at_least_warmup_no_backpay",
        "replay_updates_per_step": replay.updates_per_step, "draws": "uniform_transitions_with_replacement",
        "strategy": sampler.selection.strategy.name(), "admissibility": format!("{:?}", sampler.selection.admissibility),
        "candidate_filter": sampler.selection.candidate_filter, "relabeled_proportion": sampler.relabeled_proportion,
        "copy_update": format!("{:?}", learner.copy_update()), "truncation": format!("{:?}", replay.truncation),
        "sampler_truncation": format!("{:?}", sampler.truncation), "sampler_stream": acs2_trajectory::SAMPLER_STREAM,
        "learning_clock": "episode_end_environment_steps"})
}
