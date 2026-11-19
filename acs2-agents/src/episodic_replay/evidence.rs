use acs2_core::compaction::{subsample_indices, EvidenceSource};
use acs2_core::ports::goal::GoalLayout;
use acs2_core::support::knowledge::Transition;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::TrajectoryStore;

pub struct StoredEvidence<'a, const S: usize, const G: usize> {
    store: &'a TrajectoryStore<S, G>,
}

impl<'a, const S: usize, const G: usize> StoredEvidence<'a, S, G> {
    pub fn new(store: &'a TrajectoryStore<S, G>) -> Self {
        Self { store }
    }

    pub fn collect<const M: usize>(&self, indices: &[usize]) -> Vec<Transition<M>> {
        let mut transitions = Vec::with_capacity(indices.len());
        let mut pending = indices.iter().copied().peekable();
        let mut offset = 0usize;
        for episode in self.store.episodes() {
            let end = offset + episode.len();
            while let Some(index) = pending.next_if(|&index| index < end) {
                let local = index - offset;
                let desired = episode.start().desired;
                transitions.push(Transition::new(
                    GoalLayout::<S, G, M>::join(episode.observation(local), &desired),
                    episode.steps()[local].action,
                    GoalLayout::<S, G, M>::join(episode.observation(local + 1), &desired),
                ));
            }
            offset = end;
        }
        transitions
    }
}

impl<const S: usize, const G: usize, const M: usize> EvidenceSource<M> for StoredEvidence<'_, S, G> {
    fn sample(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<Transition<M>> {
        let indices = subsample_indices(self.store.len(), size, rng);
        self.collect::<M>(&indices)
    }
}
