#![feature(generators, generator_trait)]
#![feature(box_into_pin)]
#![feature(box_syntax)]

mod state;
mod arena;
mod nodes;
mod actions;
mod options;
mod tree_builder;
mod card_abstraction;
mod infoset;
mod cfr;

use std::ops::{Generator, GeneratorState};
use tree_builder::{TreeBuilder};
use state::{BettingRound, PlayerState, GameState};
use nodes::GameTreeNode;
use card_abstraction::{CardAbstraction, ISOMORPHIC};
use infoset::Infoset;

fn main() {
    let options = options::default_options();
    let mut tree = TreeBuilder::init(&options);
    tree.build();

    let card_abs = ISOMORPHIC::init(
            options.round,
            options.board_mask,
            &options.hand_ranges);

    // // create infosets
    // let n_rounds = 1;
    // let n_players = 2;
    // // players -> round -> action count[round] * num_hands[round]
    // let mut infosets: Vec<Vec<Vec<Infoset>>> = Vec::new();
    // for i in 0..n_players {
    //     infosets.push(Vec::new());
    //     for j in 0..n_rounds {
    //         infosets[i].push(Vec::new());
    //         for k in 0..tree.node_index[2][i] {
    //             infosets[i][j].push(Infoset::new());
    //         }
    //     }
    // }

    // let mut gen = Box::into_pin(tree.arena.generator(0));
    // loop {
    //     match gen.as_mut().resume(()) {
    //         GeneratorState::Yielded(node) => {
    //             match node {
    //                 GameTreeNode::ActionNode(an) => {
    //                     infosets[usize::from(an.player)][0][an.index] =
    //                         Infoset::init(an.actions.len() * card_abs.count);
    //                 },
    //                 _ => {}
    //             }
    //         },
    //         GeneratorState::Complete(_) => { break; }
    //     }
    // }
}
