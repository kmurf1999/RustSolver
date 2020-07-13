#![feature(test)] #![feature(generators, generator_trait)]
#![feature(box_into_pin)]
#![feature(box_syntax)]

#![allow(dead_code)]
#![allow(unused_imports)]

extern crate test;
extern crate rayon;
extern crate hashbrown;

mod constants;
mod state;
mod arena;
mod nodes;
mod action_abstraction;
mod options;
mod tree_builder;
mod card_abstraction;
mod infoset;
mod cfr;

use std::ops::{Generator, GeneratorState};
use tree_builder::{TreeBuilder};
use state::{BettingRound, PlayerState, GameState};
use infoset::{Infoset, InfosetTable};
use cfr::MCCFRTrainer;
use rand::{thread_rng};

fn main() {
    let options = options::default_river();
    let mut trainer = MCCFRTrainer::init(options);
    let mut rng = thread_rng();
    trainer.train(&mut rng, 1000);
}
