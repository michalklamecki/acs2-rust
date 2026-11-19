use acs2_agents::episodic_replay::{EpisodicAgent, EpisodicConfiguration, ReplayEnvironment};
use acs2_core::ports::environment::{Environment, StepOutcome};
use acs2_core::ports::goal::{Goal, GoalOutcome, GoalStart, GoalStep};
use acs2_core::representation::perception::Perception;
use acs2_core::support::knowledge::Transition;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::GoalEvaluator;
use serde_json::{json, Value};

use crate::runner::{CoreAgent, GoalAgent, MeasuredEnvironment, Preset};

pub mod compacting;
pub mod episodic_view;
pub mod instrumented;
pub mod vcp;

pub fn episodic_configuration(preset: Preset) -> EpisodicConfiguration {
    EpisodicConfiguration {
        capacity: preset.replay_capacity,
        warmup: preset.replay_warmup,
        updates_per_step: preset.replay_updates_per_step,
        truncation: preset.truncation,
    }
}

struct EpisodeAdapter<'a, E, const M: usize>(&'a mut E);

impl<E: Environment<M>, const M: usize> Environment<M> for EpisodeAdapter<'_, E, M> {
    fn reset(&mut self) -> Perception<M> {
        self.0.reset()
    }
    fn step(&mut self, action: usize) -> StepOutcome<M> {
        self.0.step(action)
    }
}

impl<E: MeasuredEnvironment<S, G, M>, const S: usize, const G: usize, const M: usize>
    GoalEvaluator<S, G> for EpisodeAdapter<'_, E, M>
{
    fn outcome(&self, step: &GoalStep<S, G>, goal: &Goal<G>) -> GoalOutcome {
        self.0.relabel(step, goal)
    }
    fn is_reached(&self, achieved: &Goal<G>, desired: &Goal<G>) -> bool {
        self.0.goal_reached(achieved, desired)
    }
}

impl<E: MeasuredEnvironment<S, G, M>, const S: usize, const G: usize, const M: usize>
    ReplayEnvironment<S, G, M> for EpisodeAdapter<'_, E, M>
{
    fn episode_start(&self) -> GoalStart<S, G> {
        self.0
            .episode_start()
            .expect("reset supplies a raw goal start")
    }
    fn last_step(&self) -> GoalStep<S, G> {
        self.0
            .last_transition()
            .expect("step supplies a raw goal transition")
            .step
    }
}

pub struct EpisodicGoalAgent<A>(pub A);

impl<A: EpisodicAgent<S, G, M>, const S: usize, const G: usize, const M: usize> GoalAgent<S, G, M>
    for EpisodicGoalAgent<A>
{
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        self.0
            .learner_mut()
            .train_episode(&mut EpisodeAdapter::<_, M>(env), time);
    }
    fn declared_policy(&self) -> &'static str {
        "greedy_change_anticipating_population"
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        self.0.learner().greedy_action(state, rng)
    }
    fn online_updates(&self) -> u64 {
        0
    }
    fn replay_updates(&self) -> u64 {
        episodic_view::replay_updates(self.0.learner())
    }
    fn learning_update_counts(&self) -> Option<Value> {
        Some(episodic_view::learning_update_counts(self.0.learner()))
    }
    fn replay_samples(&self) -> usize {
        0
    }
    fn replay_logical_bytes(&self) -> usize {
        0
    }
    fn trajectory_logical_bytes(&self) -> usize {
        self.0.learner().store().logical_bytes()
    }
    fn population_classifiers(&self) -> usize {
        self.0.learner().population().len()
    }
    fn population_numerosity(&self) -> u32 {
        self.0.learner().population().numerosity()
    }
    fn population_logical_bytes(&self) -> usize {
        episodic_view::population_logical_bytes(self.0.learner())
    }
    fn population_mark_entries(&self) -> usize {
        episodic_view::population_mark_entries(self.0.learner())
    }
    fn knowledge(&self, transitions: &[Transition<M>]) -> Option<f64> {
        Some(episodic_view::knowledge(self.0.learner(), transitions))
    }
    fn replay_diagnostics(&self) -> Option<Value> {
        Some(episodic_view::replay_diagnostics(self.0.learner(), self.0.name()))
    }
    fn agent_parameters(&self) -> Value {
        episodic_view::agent_parameters(self.0.learner())
    }
}

pub struct OneUpdateAdapter<R: RandomSource, const M: usize>(
    pub CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M>,
);

impl<R: RandomSource, const S: usize, const G: usize, const M: usize> GoalAgent<S, G, M>
    for OneUpdateAdapter<R, M>
{
    fn name(&self) -> &'static str {
        "acs2er_one_update"
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        self.0.train_episode(env, time)
    }
    fn declared_policy(&self) -> &'static str {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::declared_policy(
            &self.0,
        )
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::eval_action(
            &self.0, state, rng,
        )
    }
    fn online_updates(&self) -> u64 {
        0
    }
    fn replay_updates(&self) -> u64 {
        self.0.updates
    }
    fn learning_update_counts(&self) -> Option<Value> {
        Some(
            json!({"full": self.0.updates, "values_only": 0, "skipped_warmup_steps": self.0.steps - self.0.updates}),
        )
    }
    fn replay_samples(&self) -> usize {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::replay_samples(
            &self.0,
        )
    }
    fn population_classifiers(&self) -> usize {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::population_classifiers(&self.0)
    }
    fn population_numerosity(&self) -> u32 {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::population_numerosity(&self.0)
    }
    fn population_logical_bytes(&self) -> usize {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::population_logical_bytes(&self.0)
    }
    fn population_mark_entries(&self) -> usize {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::population_mark_entries(&self.0)
    }
    fn knowledge(&self, transitions: &[Transition<M>]) -> Option<f64> {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::knowledge(
            &self.0,
            transitions,
        )
    }
    fn replay_logical_bytes(&self) -> usize {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::replay_logical_bytes(&self.0) + self.0.agent.observer_logical_bytes()
    }
    fn replay_diagnostics(&self) -> Option<Value> {
        let counters = self.0.agent.counters();
        let mut diagnostics = crate::trajectory::replay_diagnostics(&counters);
        diagnostics["realized_relabeled_share"] = json!(0.0);
        diagnostics["draw_observer"] = json!("core_u1_outcomes_no_extra_random_or_objective_calls");
        Some(diagnostics)
    }
    fn agent_parameters(&self) -> Value {
        <CoreAgent<acs2_agents::acs2er::one_update::Agent<M, R>, M> as GoalAgent<S, G, M>>::agent_parameters(
            &self.0,
        )
    }
}
