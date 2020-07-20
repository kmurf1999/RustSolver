use crossbeam::sync::ShardedLock;
use crate::tree::{Tree, NodeId};
use crate::card_abstraction::{ISOMORPHIC, CardAbstraction};
use crate::nodes::GameTreeNode;

// container for infosets
pub type InfosetTable = Vec<Vec<Infoset>>;

pub fn create_infosets(n_actions: usize, tree: &Tree<GameTreeNode>, card_abs: &ISOMORPHIC) -> InfosetTable {
    let mut infosets: Vec<Vec<Infoset>> = Vec::new();

    for _ in 0..n_actions {
        infosets.push(Vec::new());
    }

    create_infosets_rec(card_abs, tree, &mut infosets, 0);

    return infosets;
}

fn create_infosets_rec(
        card_abs: &ISOMORPHIC,
        tree: &Tree<GameTreeNode>,
        infosets: &mut InfosetTable,
        node: NodeId) {
    let node = tree.get_node(node);
    match &node.data {
        GameTreeNode::Action(an) => {
            let n_actions = node.children.len();
            for _ in 0..card_abs.get_size(an.player) {
                infosets[an.index].push(Infoset::init(n_actions));
            }
            for i in 0..n_actions {
                create_infosets_rec(card_abs, tree, infosets, node.children[i]);
            }
        },
        GameTreeNode::Terminal(tn) => {},
        GameTreeNode::PrivateChance => {
            create_infosets_rec(card_abs, tree, infosets, node.children[0]);
        },
        GameTreeNode::PublicChance => {
            create_infosets_rec(card_abs, tree, infosets, node.children[0]);
        }
    }
}
    // /**
    //  * get or create infoset
    //  */
    // pub fn get_or_create(&mut self,
    //         node_idx: usize, cluster_idx: usize,
    //         n_actions: usize) -> &mut Infoset {
    //     return self.infosets[node_idx]
    //         .write()
    //         .unwrap()
    //         .entry(cluster_idx)
    //         .or_insert_with(|| Infoset::init(n_actions));
    // }

pub struct Infoset {
    pub regrets: Box<[f64]>,
    pub strategy_sum: Box<[f64]>
}

impl Infoset {
    fn new() -> Self {
        Infoset {
            regrets: Vec::new().into_boxed_slice(),
            strategy_sum: Vec::new().into_boxed_slice()
        }
    }
    pub fn init(n_actions: usize) -> Infoset {
        Infoset {
            regrets: vec![0f64; n_actions].into_boxed_slice(),
            strategy_sum: vec![0f64; n_actions].into_boxed_slice()
        }
    }
    // get strategy through regret matching
    pub fn get_strategy(&self) -> Vec<f64> {
        let n_actions = self.regrets.len();
        let mut strategy = vec![0.0; n_actions];
        let mut norm_sum = 0f64;
        for i in 0..n_actions {
            if self.regrets[i] > 0.0 {
                norm_sum += self.regrets[i];
            }
        }
        for i in 0..n_actions {
            if norm_sum > 0.0 {
                if self.regrets[i] > 0.0 {
                    strategy[i] = self.regrets[i] / norm_sum;
                }
            } else {
                strategy[i] = 1.0 / (n_actions as f64);
            }
        }
        return strategy;
    }
    pub fn get_final_strategy(&self) -> Vec<f64> {
        let n_actions = self.regrets.len();
        let mut strategy = vec![0.0; n_actions];
        let mut norm_sum = 0f64;
        for i in 0..n_actions {
            if self.strategy_sum[i] > 0.0 {
                norm_sum += self.strategy_sum[i];
            }
        }
        for i in 0..n_actions {
            if norm_sum > 0.0 {
                if self.strategy_sum[i] > 0.0 {
                    strategy[i] = self.strategy_sum[i] / norm_sum;
                }
            } else {
                strategy[i] = 1.0 / (n_actions as f64);
            }
        }
        return strategy;
    }
}
