// vectorized infoset
// to get use num_actions * cluster_idx + action
pub struct Infoset {
    regrets: Box<[u32]>,
    strategy_sum: Box<[u32]>
}

impl Infoset {
    // TODO This seems bad
    pub fn new() -> Infoset {
        Infoset {
            regrets: Box::new([]),
            strategy_sum: Box::new([]),
        }
    }
    // create MxN regrets, and strategy sum vectors
    pub fn init(size: usize) -> Infoset {
        Infoset {
            regrets: vec![0u32; size].into_boxed_slice(),
            strategy_sum: vec![0u32; size].into_boxed_slice()
        }
    }
    // get strategy through regret matching
    pub fn get_strategy(&self, cluster_idx: usize, n_actions: usize) -> Vec<u32> {
        let mut norm_sum = 0;
        let mut strategy = vec![0u32; n_actions];
        for a in 0..n_actions {
            if self.regrets[cluster_idx * n_actions + a] > 0 {
                norm_sum += self.regrets[cluster_idx * n_actions + a];
            }
        }
        for a in 0..n_actions {
            if norm_sum > 0 {
                strategy[a] = self.regrets[cluster_idx * n_actions + a] / norm_sum;
            } else {
                strategy[a] = 1000 / norm_sum;
            }
        }
        return strategy;
    }
}
