use hashbrown::HashMap;
use crate::tree_builder::TreeBuilder;

// container for infosets
pub struct InfosetTable {
    // node_idx -> map
    infosets: Vec<HashMap<usize,Infoset>>
}

impl InfosetTable {
    pub fn init(node_count: usize) -> InfosetTable {
        let mut infosets: Vec<HashMap<usize,Infoset>> = Vec::new();
        for _ in 0..node_count {
            infosets.push(HashMap::new());
        }
        InfosetTable {
            infosets
        }
    }
    /**
     * get or create infoset
     */
    pub fn get_or_create(&mut self,
            node_idx: usize, cluster_idx: usize,
            n_actions: usize) -> &mut Infoset {
        return self.infosets[node_idx]
            .entry(cluster_idx)
            .or_insert_with(|| Infoset::init(n_actions));
    }
}

pub struct Infoset {
    regrets: Box<[i32]>,
    strategy_sum: Box<[i32]>
}

impl Infoset {
    pub fn init(n_actions: usize) -> Infoset {
        Infoset {
            regrets: vec![0; n_actions].into_boxed_slice(),
            strategy_sum: vec![0; n_actions].into_boxed_slice()
        }
    }
    // get strategy through regret matching
}
