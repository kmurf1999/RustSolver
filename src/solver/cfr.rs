use rust_poker::hand::Hand;

use crate::arena::{Arena, NodeId};
use crate::infoset::Infoset;
use crate::nodes::GameTreeNode;

/**
 * A structure to implement monte carlo cfr
 */
struct MCCFR {
    // player_idx -> round_idx -> action_node_idx
    infosets: Vec<Vec<Vec<Infoset>>>,
    // public game tree
    game_tree: Arena<GameTreeNode>
}

impl MCCFR {
    /**
     * iterations: number of iterations to train for
     */
    pub fn train(&mut self, iterations: usize) {

    }
    /**
     * PARAMS:
     * node: current id of node on tree
     * player: index of player we're currently training
     * board: 64 bit representation of public cards
     * hands: 64 bit representation of hole cards
     * cfr_reach: cfr probabilility of reaching current node
     * prune: true of false, should prune
     * RETURNS:
     * utility of the currnet node for the current player
     */
    pub fn mccfr(&mut self,
            node: NodeId, player: u8,
            board: &Hand, hands: &[Hand; 2],
            cfr_reach: f64, prune: bool) -> f64 {
        let n = self.game_tree.get_node(node);
        match &n.data {
            GameTreeNode::TerminalNode(tn) => {
                // return utility for player
            },
            GameTreeNode::PrivateChanceNode(cn) => {

            },
            GameTreeNode::PublicChanceNode(cn) => {

            },
            GameTreeNode::ActionNode(an) => {

            }
        }
        return 0.0;
    }
}
