use std::iter::repeat;

use crate::arena::{Arena, NodeId};
use crate::nodes::*;
// use crate::nodes::{
//     GameTreeNode,
//     ActionNode,
//     TerminalNode, TerminalNodeType,
//     PublicChanceNode, PrivateChanceNode
// };
use crate::state::{BettingRound, GameState};
use crate::action_abstraction::{Action};
use crate::options::Options;

pub struct TreeBuilder<'a> {
    pub tree: Arena<GameTreeNode>,
    options: &'a Options,
    n_actions: usize
    // pub action_node_count: usize,
    // initial_state: GameState,
}

impl<'a> TreeBuilder<'a>{
    pub fn init(options: &'a Options) -> Self {
        TreeBuilder {
            options: options,
            tree: Arena::<GameTreeNode>::new(),
            n_actions: 0
        }
    }
    pub fn build(&mut self) {
        let initial_state = GameState::from(self.options);
        self.build_private_chance(initial_state);
    }
    pub fn action_count(&self) -> usize {
        return self.n_actions;
    }
    pub fn print(&self) {
        self.print_node(0, 0);
    }
    fn print_node(&self, node: NodeId, depth: usize) {
        let n = self.tree.get_node(node);
        let spaces = repeat("  ").take(depth).collect::<String>();
        match &n.data {
            GameTreeNode::PrivateChance => {
                println!("{}Private Chance", spaces);
                self.print_node(n.children[0], depth + 1);
            },
            GameTreeNode::PublicChance => {
                println!("{}Public Chance", spaces);
                self.print_node(n.children[0], depth + 1);
            },
            GameTreeNode::Action(an) => {
                for (i, action) in an.actions.iter().enumerate() {
                    println!("{}action: {}, player: {}",
                             spaces, action.to_string(), an.player);
                    self.print_node(n.children[i], depth + 1);
                }
            },
            GameTreeNode::Terminal(tn) => {
                println!("{}{}: {}", spaces, tn.ttype.to_string(), tn.value);
            }
        }
    }
    /**
     * build the private chance root of the tree
     */
    fn build_private_chance(&mut self, state: GameState) {

        let round = 0;
        let node = self.tree.create_node(None, GameTreeNode::PrivateChance);
        let child = self.build_action_nodes(node, round, state);
        self.tree.get_node_mut(node).add_child(child);
    }
    fn build_action_nodes(&mut self, parent: NodeId,
            round: u8, state: GameState) -> NodeId {

        let node_id = self.tree.create_node(Some(parent), GameTreeNode::Action(
            ActionNode {
                player: state.current,
                index: self.n_actions,
                round: round,
                actions: Vec::new()
            }
        ));
        self.n_actions += 1;

        match &self.tree.get_node(node_id).data {
            GameTreeNode::Action(_) => {
                for action in state.valid_actions(&self.options.action_abstraction) {
                    self.build_action(node_id, round, state, action);
                }
            },
            _ => panic!("should be action node")
        }

        return node_id
    }
    fn build_action(&mut self, node: NodeId,
                    round: u8, state: GameState, action: Action) {

        let next_state = state.apply_action(&action);
        let child;
        if next_state.bets_settled {
            if next_state.is_terminal() {
                // build terminal node
                child = self.build_terminal(node, next_state);
            } else {
                // build next round chance node
                child = self.build_public_chance(node, round, next_state.to_next_street());
            }
        } else {
            child = self.build_action_nodes(node, round, next_state);
        }

        self.tree.get_node_mut(node).add_child(child);
        match &mut self.tree.get_node_mut(node).data {
            GameTreeNode::Action(an) => {
                an.actions.push(action);
            },
            _ => panic!("must be action node")
        }
    }
    fn build_terminal(&mut self, parent: NodeId, state: GameState) -> NodeId {
        let mut terminal = TerminalNode {
            value: state.pot,
            ttype: TerminalType::SHOWDOWN,
            last_to_act: state.current
        };
        if state.is_allin() && state.round != BettingRound::River {
            terminal.ttype = TerminalType::ALLIN;
        }
        if state.is_uncontested() {
            terminal.ttype = TerminalType::UNCONTESTED;
        }
        let node = self.tree.create_node(
            Some(parent),
            GameTreeNode::Terminal(terminal));
        return node;
    }
    fn build_public_chance(&mut self, parent: NodeId, round: u8, state: GameState) -> NodeId {
        let node = self.tree.create_node(
            Some(parent),
            GameTreeNode::PublicChance);
        let child = self.build_action_nodes(node, round + 1, state);
        self.tree.get_node_mut(node).add_child(child);
        return node;
    }
}
