use acs2_core::compaction::{
    no_candidate_check, CompactionStrength, ReductionEngine, ReductionRequest, TaskGeometry,
    TriggerObservation, COMPACTION_STREAM,
};
use acs2_core::ports::goal::Goal;
use acs2_core::representation::perception::Perception;
use acs2_core::representation::population::Population;
use acs2_core::support::knowledge::Transition;
use acs2_core::support::rng::RandomSource;
use serde_json::{json, Value};

use crate::compaction_report::{record_json, CompactionReporter, CompactionView, PopulationArea};
use crate::compaction_spec::{CompactionSpecification, ShadowSpecification, SpecificationError};
use crate::runner::{GoalAgent, MeasuredEnvironment};

pub trait Inspected<const S: usize, const G: usize, const M: usize> {
    fn head_populations(&self) -> Vec<&Population<M>>;
    fn interactions(&self) -> u64;
    fn evidence(&self, size: usize) -> Option<Vec<Transition<M>>>;
    fn compaction_view(&self) -> Option<CompactionView<'_>> {
        None
    }
}

#[derive(Clone, Copy, Default)]
pub struct Instrumentation {
    pub name: Option<&'static str>,
    pub compaction: Option<&'static CompactionSpecification>,
    pub shadow: Option<&'static ShadowSpecification>,
}

impl Instrumentation {
    pub fn is_empty(&self) -> bool {
        self.name.is_none() && self.compaction.is_none() && self.shadow.is_none()
    }
}

struct CompactionInstrument {
    specification: &'static CompactionSpecification,
    reporter: CompactionReporter,
}

struct ShadowEngines<const M: usize> {
    label: String,
    strength: CompactionStrength,
    heads: Vec<ReductionEngine<M>>,
}

struct ShadowInstrument<const M: usize> {
    specification: &'static ShadowSpecification,
    entries: Vec<ShadowEngines<M>>,
}

struct NoRandomness;

impl RandomSource for NoRandomness {
    fn gen_bool(&mut self, _probability: f64) -> bool {
        unreachable!("shadow evidence is strided and draws no randomness")
    }
    fn gen_range(&mut self, _bound: usize) -> usize {
        unreachable!("shadow evidence is strided and draws no randomness")
    }
    fn gen_unit(&mut self) -> f64 {
        unreachable!("shadow evidence is strided and draws no randomness")
    }
}

pub struct Instrumented<A, const M: usize> {
    pub inner: A,
    name: Option<&'static str>,
    compaction: Option<CompactionInstrument>,
    shadow: Option<ShadowInstrument<M>>,
    area: PopulationArea,
}

impl<A, const M: usize> Instrumented<A, M> {
    pub fn new<const S: usize, const G: usize>(
        inner: A,
        instrumentation: Instrumentation,
        geometry: &TaskGeometry,
    ) -> Result<Self, SpecificationError>
    where
        A: Inspected<S, G, M>,
    {
        let compaction = instrumentation.compaction.map(|specification| CompactionInstrument {
            specification,
            reporter: CompactionReporter::new(specification.max_records_per_row),
        });
        let shadow = match instrumentation.shadow {
            None => None,
            Some(specification) => {
                let entries = specification.entries()?;
                if specification.evidence_sample > 0 && inner.evidence(1).is_none() {
                    return Err(SpecificationError::new(
                        "this agent keeps no experience memory, so a shadow run on it needs evidence_sample 0 and no evidence operators",
                    ));
                }
                let heads = inner.head_populations().len();
                Some(ShadowInstrument {
                    specification,
                    entries: entries
                        .into_iter()
                        .map(|entry| ShadowEngines {
                            label: entry.label,
                            strength: entry.strength,
                            heads: (0..heads)
                                .map(|_| ReductionEngine::new(entry.parameters.clone(), geometry.clone()))
                                .collect(),
                        })
                        .collect(),
                })
            }
        };
        Ok(Self {
            inner,
            name: instrumentation.name,
            compaction,
            shadow,
            area: PopulationArea::default(),
        })
    }
}

impl<
        A: GoalAgent<S, G, M> + Inspected<S, G, M>,
        const S: usize,
        const G: usize,
        const M: usize,
    > GoalAgent<S, G, M> for Instrumented<A, M>
{
    fn name(&self) -> &'static str {
        self.name.unwrap_or_else(|| self.inner.name())
    }
    fn register_training_goals(&mut self, goals: &[Goal<G>]) {
        self.inner.register_training_goals(goals)
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        let before = env.measured_steps();
        self.inner.train_episode(env, time);
        self.area.observe_episode(
            before,
            env.measured_steps(),
            self.inner.population_classifiers(),
        );
    }
    fn declared_policy(&self) -> &'static str {
        self.inner.declared_policy()
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        self.inner.eval_action(state, rng)
    }
    fn evaluation_heads(&self) -> usize {
        self.inner.evaluation_heads()
    }
    fn eval_head_action(
        &self,
        head: usize,
        state: &Perception<M>,
        rng: &mut dyn RandomSource,
    ) -> (usize, f64) {
        self.inner.eval_head_action(head, state, rng)
    }
    fn online_updates(&self) -> u64 {
        self.inner.online_updates()
    }
    fn replay_updates(&self) -> u64 {
        self.inner.replay_updates()
    }
    fn learning_update_counts(&self) -> Option<Value> {
        self.inner.learning_update_counts()
    }
    fn replay_samples(&self) -> usize {
        self.inner.replay_samples()
    }
    fn population_classifiers(&self) -> usize {
        self.inner.population_classifiers()
    }
    fn population_numerosity(&self) -> u32 {
        self.inner.population_numerosity()
    }
    fn population_logical_bytes(&self) -> usize {
        self.inner.population_logical_bytes()
    }
    fn population_mark_entries(&self) -> usize {
        self.inner.population_mark_entries()
    }
    fn knowledge(&self, transitions: &[Transition<M>]) -> Option<f64> {
        self.inner.knowledge(transitions)
    }
    fn replay_logical_bytes(&self) -> usize {
        self.inner.replay_logical_bytes()
    }
    fn trajectory_logical_bytes(&self) -> usize {
        self.inner.trajectory_logical_bytes()
    }
    fn replay_diagnostics(&self) -> Option<Value> {
        self.inner.replay_diagnostics()
    }
    fn agent_parameters(&self) -> Value {
        let mut parameters = self.inner.agent_parameters();
        if let Some(compaction) = &self.compaction {
            parameters["compaction"] = compaction.specification.echo();
            parameters["compaction_stream"] = json!(COMPACTION_STREAM);
        }
        parameters
    }
    fn ensemble_diagnostics(&self, transitions: Option<&[Transition<M>]>) -> Option<Value> {
        self.inner.ensemble_diagnostics(transitions)
    }
    fn head_disagreement(&self, transitions: &[Transition<M>]) -> Option<Value> {
        self.inner.head_disagreement(transitions)
    }
    fn compaction_diagnostics(&mut self) -> Option<Value> {
        let compaction = self.compaction.as_mut()?;
        let view = self.inner.compaction_view()?;
        Some(compaction.reporter.report(&view, &self.area))
    }
    fn compaction_shadow_diagnostics(&mut self) -> Option<Value> {
        let shadow = self.shadow.as_mut()?;
        let specification = shadow.specification;
        let populations = self.inner.head_populations();
        let evidence = if specification.evidence_sample > 0 {
            self.inner
                .evidence(specification.evidence_sample)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let interactions = self.inner.interactions();
        let mut random = NoRandomness;
        let mut entries = Vec::with_capacity(shadow.entries.len());
        for entry in &mut shadow.entries {
            let mut heads = Vec::with_capacity(populations.len());
            for (head, (engine, population)) in
                entry.heads.iter_mut().zip(&populations).enumerate()
            {
                let request = ReductionRequest {
                    interaction: interactions,
                    observation: TriggerObservation {
                        interactions,
                        population_size: population.len(),
                        ..TriggerObservation::default()
                    },
                    strength: entry.strength,
                };
                engine.reduce(
                    &request,
                    population,
                    &evidence[..],
                    &mut random,
                    &mut no_candidate_check,
                );
                let record = engine.records().last().expect("a reduction was just recorded");
                heads.push(record_json(head, record));
            }
            entries.push(json!({"label": entry.label, "heads": heads}));
        }
        Some(json!({"specification": specification.echo(), "interactions": interactions,
            "evidence_size": evidence.len(),
            "evidence_definition": "strided_sample_of_the_agent_memory_no_random_stream",
            "population_size_area": self.area.json(),
            "entries": entries}))
    }
}
