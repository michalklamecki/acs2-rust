use crate::support::knowledge::Transition;
use crate::support::rng::RandomSource;

pub trait EvidenceSource<const N: usize> {
    fn sample(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<Transition<N>>;
}

pub struct NoEvidence;

impl<const N: usize> EvidenceSource<N> for NoEvidence {
    fn sample(&self, _size: usize, _rng: &mut dyn RandomSource) -> Vec<Transition<N>> {
        Vec::new()
    }
}

impl<const N: usize> EvidenceSource<N> for [Transition<N>] {
    fn sample(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<Transition<N>> {
        subsample_indices(self.len(), size, rng)
            .into_iter()
            .map(|index| {
                let transition = &self[index];
                Transition::new(transition.p0, transition.action, transition.p1)
            })
            .collect()
    }
}

pub fn subsample_indices(total: usize, size: usize, rng: &mut dyn RandomSource) -> Vec<usize> {
    if size >= total {
        return (0..total).collect();
    }
    let mut pool: Vec<usize> = (0..total).collect();
    for position in 0..size {
        let picked = position + rng.gen_range(total - position);
        pool.swap(position, picked);
    }
    pool.truncate(size);
    pool.sort_unstable();
    pool
}

pub fn strided_indices(total: usize, size: usize) -> Vec<usize> {
    if size >= total {
        return (0..total).collect();
    }
    (0..size).map(|position| position * total / size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::test_support::all_transitions;
    use crate::support::rng::ChaChaRandomSource;

    struct Forbidden;

    impl RandomSource for Forbidden {
        fn gen_bool(&mut self, _: f64) -> bool {
            panic!("randomness consumed")
        }
        fn gen_range(&mut self, _: usize) -> usize {
            panic!("randomness consumed")
        }
        fn gen_unit(&mut self) -> f64 {
            panic!("randomness consumed")
        }
    }

    #[test]
    fn a_sample_not_smaller_than_the_pool_takes_all_without_randomness() {
        assert_eq!(subsample_indices(5, 5, &mut Forbidden), vec![0, 1, 2, 3, 4]);
        assert_eq!(subsample_indices(5, 9, &mut Forbidden), vec![0, 1, 2, 3, 4]);
        assert_eq!(subsample_indices(0, 3, &mut Forbidden), Vec::<usize>::new());
        assert_eq!(subsample_indices(4, 0, &mut ChaChaRandomSource::from_seed(1)), Vec::<usize>::new());
    }

    #[test]
    fn a_subsample_is_sorted_distinct_and_seeded() {
        let draw = |seed| subsample_indices(40, 9, &mut ChaChaRandomSource::from_seed(seed));
        let first = draw(7);
        assert_eq!(first, draw(7));
        assert_ne!(first, draw(8));
        assert_eq!(first.len(), 9);
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(first.iter().all(|&index| index < 40));
    }

    #[test]
    fn strided_indices_are_spread_and_need_no_randomness() {
        assert_eq!(strided_indices(10, 5), vec![0, 2, 4, 6, 8]);
        assert_eq!(strided_indices(3, 5), vec![0, 1, 2]);
    }

    #[test]
    fn a_slice_samples_its_own_transitions() {
        let pool = all_transitions();
        let sampled = pool[..].sample(8, &mut ChaChaRandomSource::from_seed(3));
        assert_eq!(sampled.len(), 8);
        let everything = pool[..].sample(pool.len(), &mut Forbidden);
        assert_eq!(everything.len(), pool.len());
        let empty: Vec<Transition<3>> = NoEvidence.sample(4, &mut Forbidden);
        assert!(empty.is_empty());
    }
}
