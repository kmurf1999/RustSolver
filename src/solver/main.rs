#![feature(test)]
#![feature(generators, generator_trait)]
#![feature(box_into_pin)]
#![feature(box_syntax)]

#![allow(dead_code)]
#![allow(unused_imports)]

extern crate test;
extern crate rayon;
extern crate crossbeam;
extern crate hashbrown;

mod constants;
mod state;
mod tree;
mod nodes;
mod action_abstraction;
mod options;
mod tree_builder;
mod card_abstraction;
mod infoset;
mod cfr;

use cfr::MCCFRTrainer;
use std::time::Instant;

fn main() {
    let options = options::default_river();
    let mut trainer = MCCFRTrainer::init(options);
    let start = Instant::now();
    trainer.train(50_000_000);
    let elapsed = start.elapsed().subsec_nanos();
    println!("{}", elapsed);
}
