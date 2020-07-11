use std::iter::repeat;

use crate::arena::{Arena, NodeId};
use crate::nodes::{
    GameTreeNode,
    ActionNode,
    TerminalNode, TerminalNodeType,
    PublicChanceNode, PrivateChanceNode
};
use crate::state::{BettingRound, GameState};
use crate::actions::{Action, ACTIONS};
use crate::options::Options;

const MAX_PLAYERS: usize = 2;
const MAX_ROUNDS: usize = 3; // flop, turn, river

pub struct TreeBuilder {
    pub arena: Arena<GameTreeNode>,
    initial_state: GameState,
    // action node indices for each player on each round
    pub node_index: [[usize; MAX_PLAYERS]; MAX_ROUNDS]
}

impl TreeBuilder {
    pub fn init(options: &Options) -> TreeBuilder {
        TreeBuilder {
            arena: Arena::<GameTreeNode>::new(),
            initial_state: options.to_state(),
            node_index: [[0; MAX_PLAYERS]; MAX_ROUNDS]
        }
    }
    pub fn build(&mut self) {
        self.build_private_chance_node(0, self.initial_state);
    }
    fn build_action_nodes(&mut self, parent: NodeId, state: GameState) -> NodeId {
        let node = self.arena.create_node(GameTreeNode::ActionNode(
            ActionNode {
                index: self.node_index[state.round.to_usize()]
                    [usize::from(state.current)],
                player: state.current,
                actions: Vec::new()
            }
        ));

        self.node_index[state.round.to_usize()][usize::from(state.current)] += 1;

        // link to parent
        self.arena.get_node_mut(node).set_parent(parent);

        for action in ACTIONS.iter() {
            self.build_action(node, action, state);
        }

        return node;
    }
    fn build_action(&mut self, node: NodeId, action: &Action, state: GameState) {
        if state.is_valid_action(action) {
            let next_state = state.apply_action(action);
            if next_state.bets_settled {
                // if fold, allin, or showdown...
                if next_state.round == BettingRound::River
                        || next_state.is_allin() 
                        || next_state.is_uncontested() {
                    // build terminal node
                    let child = self.build_terminal_node(node, next_state);
                    self.arena.get_node_mut(node).add_child(child);
                } else {
                    // build next round chance node
                    let child = self.build_public_chance_node(node, next_state.to_next_street());
                    self.arena.get_node_mut(node).add_child(child);
                }
            } else {
                // build more action nodes
                let child = self.build_action_nodes(node, next_state);
                self.arena.get_node_mut(node).add_child(child);
            }
            // add action
            match &mut self.arena.get_node_mut(node).data {
                GameTreeNode::ActionNode(action_node) => {
                    action_node.actions.push(action.clone());
                },
                _ => {
                    // throw error
                    assert!(false);
                }
            }
        }
    }
    fn build_terminal_node(&mut self, parent: NodeId, state: GameState) -> NodeId {
        let mut terminal = TerminalNode {
            value: state.pot / 2,
            node_type: TerminalNodeType::Showdown,
            last_to_act: state.current,
            board_mask: state.board_mask
        };
        if state.is_allin() {
            terminal.node_type = TerminalNodeType::Allin
        }
        if state.is_uncontested() {
            terminal.node_type = TerminalNodeType::Uncontested
        }
        let node = self.arena.create_node(
            GameTreeNode::TerminalNode(terminal));

        // link to parent
        self.arena.get_node_mut(node).set_parent(parent);
        return node;
    }
    fn build_private_chance_node(&mut self, parent: NodeId, state: GameState) -> NodeId {
        let chance = PrivateChanceNode {
            board_mask: state.board_mask
        };
        let node = self.arena.create_node(GameTreeNode::PrivateChanceNode(chance));

        let child = self.build_action_nodes(node, state);
        self.arena.get_node_mut(node).add_child(child);
        self.arena.get_node_mut(node).set_parent(parent);
        return node;
    }
    fn build_public_chance_node(&mut self, parent: NodeId, state: GameState) -> NodeId {
        let chance = PublicChanceNode {
            board_mask: state.board_mask
        };
        let node = self.arena.create_node(GameTreeNode::PublicChanceNode(chance));

        let child = self.build_action_nodes(node, state);
        self.arena.get_node_mut(node).add_child(child);
        self.arena.get_node_mut(node).set_parent(parent);
        return node;
    }
    pub fn print_node(&self, node: NodeId, depth: usize) {
        let n = self.arena.get_node(node);
        let spaces = repeat(' ').take(depth).collect::<String>();
        match &n.data {
            GameTreeNode::ActionNode(an) => {
                for (i, action) in an.actions.iter().enumerate() {
                    println!("{}{} - player: {}, index: {}", spaces, action.to_string(), an.player, an.index);
                    self.print_node(n.children[i], depth + 2);
                }
            },
            GameTreeNode::TerminalNode(tn) => {
                match &tn.node_type {
                    TerminalNodeType::Allin => {
                        println!("{}Allin - value: {}", spaces, tn.value);
                    },
                    TerminalNodeType::Showdown => {
                        println!("{}Showdown - value: {}", spaces, tn.value);
                    },
                    TerminalNodeType::Uncontested => {
                        println!("{}Uncontested - value: {}", spaces, tn.value);
                    }
                }
            },
            GameTreeNode::PrivateChanceNode(_) => {
                println!("{}Private Chance", spaces);
                self.print_node(n.children[0], depth + 2);
            },
            GameTreeNode::PublicChanceNode(_) => {
                println!("{}Public Chance", spaces);
                self.print_node(n.children[0], depth + 2);
            }
        }
    }
}
