use std::mem::size_of;

use acs2_agents::acs2vcp::compacting::{CompactingAgent as VcpCompactingAgent, CompactionClock};
use acs2_agents::acs2vcp::Preset as VcpPreset;
use acs2_agents::episodic_replay::compacting::CompactingEpisodic;
use acs2_agents::episodic_replay::evidence::StoredEvidence;
use acs2_agents::episodic_replay::EpisodicAgent;
use acs2_core::compaction::{strided_indices, CompactionController};
use acs2_core::representation::classifier::Classifier;
use acs2_core::representation::perception::Perception;
use acs2_core::representation::population::Population;
use acs2_core::support::knowledge::Transition;
use acs2_core::support::rng::RandomSource;
use acs2_core::trial::LearningAgent;
use acs2_trajectory::TrajectoryStore;
use serde_json::Value;

use super::instrumented::Inspected;
use super::vcp::{record_avoided_work, TrainingDisagreement, VcpGoalAgent, VcpView};
use super::{episodic_view, EpisodeAdapter, OneUpdateAdapter};
use crate::compaction_report::{CompactionView, HeadCompaction, VcpCompaction};
use crate::runner::{CoreAgent, GoalAgent, MeasuredEnvironment};

pub trait LearnerInspection<const M: usize> {
    fn memory_evidence(&self, size: usize) -> Option<Vec<Transition<M>>>;

    fn compaction_view(&self) -> Option<CompactionView<'_>> {
        None
    }
}

fn single_head_view<const N: usize>(
    controller: &CompactionController<N>,
    cost: acs2_core::compaction::MatchingCost,
) -> CompactionView<'_> {
    CompactionView {
        heads: vec![HeadCompaction {
            records: controller.records(),
            cost,
            policy: Some(controller.policy()),
            skips: Some(controller.skips()),
        }],
        synchronous_policy: None,
        vcp: None,
    }
}

fn memory_evidence<const M: usize>(
    memory: &acs2_agents::acs2er::replay::ReplayMemory<M>,
    size: usize,
) -> Vec<Transition<M>> {
    strided_indices(memory.len(), size)
        .into_iter()
        .map(|index| {
            let sample = memory.get(index);
            Transition::new(sample.state, sample.action, sample.next_state)
        })
        .collect()
}

fn stored_evidence<const S: usize, const G: usize, const M: usize>(
    store: &TrajectoryStore<S, G>,
    size: usize,
) -> Vec<Transition<M>> {
    StoredEvidence::new(store).collect::<M>(&strided_indices(store.len(), size))
}

impl<const M: usize, R: RandomSource> LearnerInspection<M> for acs2_agents::acs2::Agent<M, R> {
    fn memory_evidence(&self, _size: usize) -> Option<Vec<Transition<M>>> {
        None
    }
}

impl<const M: usize, R: RandomSource> LearnerInspection<M> for acs2_agents::acs2er::Agent<M, R> {
    fn memory_evidence(&self, size: usize) -> Option<Vec<Transition<M>>> {
        Some(memory_evidence(self.replay_memory(), size))
    }
}

impl<const M: usize, R: RandomSource> LearnerInspection<M>
    for acs2_agents::acs2er::one_update::Agent<M, R>
{
    fn memory_evidence(&self, _size: usize) -> Option<Vec<Transition<M>>> {
        None
    }
}

impl<const M: usize, R: RandomSource, C: RandomSource> LearnerInspection<M>
    for acs2_agents::acs2::compacting::CompactingAgent<M, R, C>
{
    fn memory_evidence(&self, _size: usize) -> Option<Vec<Transition<M>>> {
        None
    }

    fn compaction_view(&self) -> Option<CompactionView<'_>> {
        Some(single_head_view(self.controller(), self.cost()))
    }
}

impl<const M: usize, R: RandomSource, C: RandomSource> LearnerInspection<M>
    for acs2_agents::acs2er::compacting::CompactingAgent<M, R, C>
{
    fn memory_evidence(&self, size: usize) -> Option<Vec<Transition<M>>> {
        Some(memory_evidence(self.agent().replay_memory(), size))
    }

    fn compaction_view(&self) -> Option<CompactionView<'_>> {
        Some(single_head_view(self.controller(), self.cost()))
    }
}

impl<A, const S: usize, const G: usize, const M: usize> Inspected<S, G, M> for CoreAgent<A, M>
where
    A: LearningAgent<M> + LearnerInspection<M>,
{
    fn head_populations(&self) -> Vec<&Population<M>> {
        vec![self.agent.population()]
    }

    fn interactions(&self) -> u64 {
        self.steps
    }

    fn evidence(&self, size: usize) -> Option<Vec<Transition<M>>> {
        self.agent.memory_evidence(size)
    }

    fn compaction_view(&self) -> Option<CompactionView<'_>> {
        LearnerInspection::compaction_view(&self.agent)
    }
}

impl<R: RandomSource, const S: usize, const G: usize, const M: usize> Inspected<S, G, M>
    for OneUpdateAdapter<R, M>
{
    fn head_populations(&self) -> Vec<&Population<M>> {
        <CoreAgent<_, M> as Inspected<S, G, M>>::head_populations(&self.0)
    }

    fn interactions(&self) -> u64 {
        <CoreAgent<_, M> as Inspected<S, G, M>>::interactions(&self.0)
    }

    fn evidence(&self, size: usize) -> Option<Vec<Transition<M>>> {
        <CoreAgent<_, M> as Inspected<S, G, M>>::evidence(&self.0, size)
    }
}

impl<A: EpisodicAgent<S, G, M>, const S: usize, const G: usize, const M: usize>
    Inspected<S, G, M> for super::EpisodicGoalAgent<A>
{
    fn head_populations(&self) -> Vec<&Population<M>> {
        vec![self.0.learner().population()]
    }

    fn interactions(&self) -> u64 {
        self.0.learner().counts().environment_steps
    }

    fn evidence(&self, size: usize) -> Option<Vec<Transition<M>>> {
        Some(stored_evidence(self.0.learner().store(), size))
    }
}

pub struct CompactingEpisodicGoalAgent<const S: usize, const G: usize, const M: usize, A, C>(
    pub CompactingEpisodic<S, G, M, A, C>,
);

impl<const S: usize, const G: usize, const M: usize, A, C> GoalAgent<S, G, M>
    for CompactingEpisodicGoalAgent<S, G, M, A, C>
where
    A: EpisodicAgent<S, G, M>,
    C: RandomSource,
{
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        self.0.train_episode(&mut EpisodeAdapter::<_, M>(env), time);
    }
    fn declared_policy(&self) -> &'static str {
        "greedy_change_anticipating_population"
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        self.0.agent().learner().greedy_action(state, rng)
    }
    fn online_updates(&self) -> u64 {
        0
    }
    fn replay_updates(&self) -> u64 {
        episodic_view::replay_updates(self.0.agent().learner())
    }
    fn learning_update_counts(&self) -> Option<Value> {
        Some(episodic_view::learning_update_counts(self.0.agent().learner()))
    }
    fn replay_samples(&self) -> usize {
        0
    }
    fn replay_logical_bytes(&self) -> usize {
        0
    }
    fn trajectory_logical_bytes(&self) -> usize {
        self.0.agent().learner().store().logical_bytes()
    }
    fn population_classifiers(&self) -> usize {
        self.0.population().len()
    }
    fn population_numerosity(&self) -> u32 {
        self.0.population().numerosity()
    }
    fn population_logical_bytes(&self) -> usize {
        episodic_view::population_logical_bytes(self.0.agent().learner())
    }
    fn population_mark_entries(&self) -> usize {
        episodic_view::population_mark_entries(self.0.agent().learner())
    }
    fn knowledge(&self, transitions: &[Transition<M>]) -> Option<f64> {
        Some(episodic_view::knowledge(self.0.agent().learner(), transitions))
    }
    fn replay_diagnostics(&self) -> Option<Value> {
        Some(episodic_view::replay_diagnostics(
            self.0.agent().learner(),
            self.0.name(),
        ))
    }
    fn agent_parameters(&self) -> Value {
        episodic_view::agent_parameters(self.0.agent().learner())
    }
}

impl<const S: usize, const G: usize, const M: usize, A, C> Inspected<S, G, M>
    for CompactingEpisodicGoalAgent<S, G, M, A, C>
where
    A: EpisodicAgent<S, G, M>,
    C: RandomSource,
{
    fn head_populations(&self) -> Vec<&Population<M>> {
        vec![self.0.population()]
    }

    fn interactions(&self) -> u64 {
        self.0.agent().learner().counts().environment_steps
    }

    fn evidence(&self, size: usize) -> Option<Vec<Transition<M>>> {
        Some(stored_evidence(self.0.agent().learner().store(), size))
    }

    fn compaction_view(&self) -> Option<CompactionView<'_>> {
        Some(single_head_view(self.0.controller(), self.0.cost()))
    }
}

impl<
        const S: usize,
        const G: usize,
        const M: usize,
        R: RandomSource,
        P: RandomSource,
        A: RandomSource,
    > Inspected<S, G, M> for VcpGoalAgent<S, G, M, R, P, A>
{
    fn head_populations(&self) -> Vec<&Population<M>> {
        self.agent.heads().iter().map(|head| head.population()).collect()
    }

    fn interactions(&self) -> u64 {
        self.agent.counts().environment_steps
    }

    fn evidence(&self, size: usize) -> Option<Vec<Transition<M>>> {
        Some(stored_evidence(self.agent.store(), size))
    }
}

pub struct CompactingVcpGoalAgent<const S: usize, const G: usize, const M: usize, R, P, A, C> {
    pub agent: VcpCompactingAgent<S, G, M, R, P, A, C>,
    pub preset: VcpPreset,
    pub training_disagreement: TrainingDisagreement,
}

impl<
        const S: usize,
        const G: usize,
        const M: usize,
        R: RandomSource,
        P: RandomSource,
        A: RandomSource,
        C: RandomSource,
    > CompactingVcpGoalAgent<S, G, M, R, P, A, C>
{
    pub fn snapshot(&self) -> String {
        format!("{}{:?}", self.agent.snapshot(), self.training_disagreement)
    }

    fn view(&self) -> VcpView<'_, S, G, M, R, P, A> {
        VcpView {
            agent: self.agent.agent(),
            training_disagreement: &self.training_disagreement,
            measurement_inline_bytes: size_of::<Self>()
                - size_of::<acs2_agents::acs2vcp::Agent<S, G, M, R, P, A>>(),
        }
    }
}

impl<
        const S: usize,
        const G: usize,
        const M: usize,
        R: RandomSource,
        P: RandomSource,
        A: RandomSource,
        C: RandomSource,
    > GoalAgent<S, G, M> for CompactingVcpGoalAgent<S, G, M, R, P, A, C>
{
    fn name(&self) -> &'static str {
        self.preset.name()
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        let proposals_before = self.agent.agent().priority_counts().proposals;
        let classifiers_before = self.population_classifiers();
        let diagnostic = &mut self.training_disagreement;
        self.agent.train_episode_observed(
            &mut EpisodeAdapter::<_, M>(env),
            time,
            &mut |variance, accepted| diagnostic.record(variance, accepted),
        );
        record_avoided_work(
            &mut self.training_disagreement,
            self.agent.agent(),
            proposals_before,
            classifiers_before,
        );
    }
    fn declared_policy(&self) -> &'static str {
        self.view().declared_policy()
    }
    fn evaluation_heads(&self) -> usize {
        self.agent.agent().heads().len()
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        assert_eq!(
            self.evaluation_heads(),
            1,
            "ensemble evaluation must name its head"
        );
        self.agent.agent().greedy_action(0, state, rng)
    }
    fn eval_head_action(
        &self,
        head: usize,
        state: &Perception<M>,
        rng: &mut dyn RandomSource,
    ) -> (usize, f64) {
        self.agent.agent().greedy_action(head, state, rng)
    }
    fn online_updates(&self) -> u64 {
        0
    }
    fn replay_updates(&self) -> u64 {
        let counts = self.agent.agent().counts();
        counts.full + counts.values_only
    }
    fn learning_update_counts(&self) -> Option<Value> {
        let counts = self.agent.agent().counts();
        Some(
            serde_json::json!({"full": counts.full, "values_only": counts.values_only,
                "skipped_warmup_steps": counts.skipped_warmup_steps}),
        )
    }
    fn replay_samples(&self) -> usize {
        0
    }
    fn replay_logical_bytes(&self) -> usize {
        0
    }
    fn trajectory_logical_bytes(&self) -> usize {
        self.agent.agent().store().logical_bytes()
    }
    fn population_classifiers(&self) -> usize {
        self.view().population_classifiers()
    }
    fn population_numerosity(&self) -> u32 {
        self.view().population_numerosity()
    }
    fn population_logical_bytes(&self) -> usize {
        self.population_classifiers() * size_of::<Classifier<M>>()
    }
    fn population_mark_entries(&self) -> usize {
        self.view().population_mark_entries()
    }
    fn knowledge(&self, transitions: &[Transition<M>]) -> Option<f64> {
        Some(self.view().knowledge(transitions))
    }
    fn head_disagreement(&self, transitions: &[Transition<M>]) -> Option<Value> {
        self.view().head_disagreement(transitions)
    }
    fn ensemble_diagnostics(&self, transitions: Option<&[Transition<M>]>) -> Option<Value> {
        Some(self.view().ensemble_diagnostics(transitions))
    }
    fn replay_diagnostics(&self) -> Option<Value> {
        Some(self.view().replay_diagnostics())
    }
    fn agent_parameters(&self) -> Value {
        self.view().agent_parameters()
    }
}

impl<
        const S: usize,
        const G: usize,
        const M: usize,
        R: RandomSource,
        P: RandomSource,
        A: RandomSource,
        C: RandomSource,
    > Inspected<S, G, M> for CompactingVcpGoalAgent<S, G, M, R, P, A, C>
{
    fn head_populations(&self) -> Vec<&Population<M>> {
        self.agent
            .agent()
            .heads()
            .iter()
            .map(|head| head.population())
            .collect()
    }

    fn interactions(&self) -> u64 {
        self.agent.agent().counts().environment_steps
    }

    fn evidence(&self, size: usize) -> Option<Vec<Transition<M>>> {
        Some(stored_evidence(self.agent.agent().store(), size))
    }

    fn compaction_view(&self) -> Option<CompactionView<'_>> {
        let heads = self.agent.agent().heads().len();
        let (clock, per_head) = match self.agent.configuration().clock() {
            CompactionClock::PerHead => ("per_head", true),
            CompactionClock::Synchronous => ("synchronous", false),
        };
        let controllers = self.agent.head_controllers();
        let head_views = (0..heads)
            .map(|head| HeadCompaction {
                records: self.agent.reduction_records(head),
                cost: self.agent.head_cost(head),
                policy: controllers
                    .filter(|_| per_head)
                    .map(|controllers| controllers[head].policy()),
                skips: controllers
                    .filter(|_| per_head)
                    .map(|controllers| controllers[head].skips()),
            })
            .collect();
        Some(CompactionView {
            heads: head_views,
            synchronous_policy: self.agent.synchronous_policy(),
            vcp: Some(VcpCompaction {
                clock,
                acceptance: self.agent.acceptance_audits(),
                probes: self.agent.disagreement_records(),
                acceptance_work: self.agent.acceptance_work(),
                probe_work: self.agent.probe_work(),
            }),
        })
    }
}
