use std::iter::repeat;
use crate::tree::{Tree, NodeId};
use crate::nodes::*;
use crate::state::{BettingRound, GameState};
use crate::action_abstraction::{Action};
use crate::options::Options;


pub fn build_game_tree(options: &Options) -> (usize, Tree<GameTreeNode>) {
    let mut builder = TreeBuilder::init(options);
    let initial_state = GameState::from(options);
    builder.build_private_chance(initial_state);
    return (builder.n_actions, builder.tree);
}

struct TreeBuilder<'a> {
    tree: Tree<GameTreeNode>,
    options: &'a Options,
    n_actions: usize
}

impl<'a> TreeBuilder<'a>{
    fn init(options: &'a Options) -> Self {
        TreeBuilder {
            options: options,
            tree: Tree::<GameTreeNode>::new(),
            n_actions: 0
        }
    }
    fn print(&self) {
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
            GameTreeNode::PublicChance(_) => {
                println!("{}Public Chance", spaces);
                self.print_node(n.children[0], depth + 1);
            },
            GameTreeNode::Action(an) => {
                for (i, action) in an.actions.iter().enumerate() {
                    println!("{}action: {}, idx: {} player: {}",
                             spaces, action.to_string(), an.index, an.player);
                    self.print_node(n.children[i], depth + 1);
                }
            },
            GameTreeNode::Terminal(tn) => {
                println!("{}{}: last to act {} {}", spaces, tn.ttype.to_string(), tn.last_to_act, tn.value);
            }
        }
    }
    /**
     * build the private chance root of the tree
     */
    fn build_private_chance(&mut self, state: GameState) {

        let round_idx = 0;
        let node = self.tree.create_node(None, GameTreeNode::PrivateChance);
        let child = self.build_action_nodes(node, round_idx, state);
        self.tree.get_node_mut(node).add_child(child);
    }
    fn build_action_nodes(&mut self, parent: NodeId,
            round_idx: u8, state: GameState) -> NodeId {

        let node_id = self.tree.create_node(Some(parent), GameTreeNode::Action(
            ActionNode {
                player: state.current,
                index: self.n_actions,
                round_idx: round_idx,
                actions: Vec::new()
            }
        ));
        self.n_actions += 1;

        match &self.tree.get_node(node_id).data {
            GameTreeNode::Action(_) => {
                for action in state.valid_actions(&self.options.action_abstraction, round_idx.into()) {
                    self.build_action(node_id, round_idx, state, action);
                }
            },
            _ => panic!("should be action node")
        }

        return node_id
    }
    fn build_action(&mut self, node: NodeId,
                    round_idx: u8, state: GameState, action: Action) {

        let next_state = state.apply_action(&action);
        let child;
        if next_state.bets_settled {
            if next_state.is_terminal() {
                // build terminal node
                child = self.build_terminal(node, next_state);
            } else {
                // build next round chance node
                child = self.build_public_chance(node, round_idx, next_state.to_next_street());
            }
        } else {
            child = self.build_action_nodes(node, round_idx, next_state);
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
            last_to_act: state.current,
            round: state.round,
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
    fn build_public_chance(&mut self, parent: NodeId, round_idx: u8, state: GameState) -> NodeId {
        let node = self.tree.create_node(
            Some(parent),
            GameTreeNode::PublicChance(PublicChanceNode {
                round: state.round
            }));
        let child = self.build_action_nodes(node, round_idx + 1, state);
        self.tree.get_node_mut(node).add_child(child);
        return node;
    }
}
