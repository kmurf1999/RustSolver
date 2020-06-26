extern crate bytepack;
extern crate rust_poker;

use rust_solver::hand_index::{ hand_indexer_t, hand_index_t };

use bytepack::{ LEUnpacker };
use std::io;
use std::io::prelude::*;
use std::fs::File;
use std::io::SeekFrom;

use rust_poker::card_range::{ get_card_mask, RANKS, SUITS };

struct Indexes {
    indexer: hand_indexer_t,
    pub offsets: [u64; 4]
}

impl Indexes {
    pub fn new() -> Indexes {
        let indexer = hand_indexer_t::init(4, [ 2, 3, 1, 1, ].to_vec());
        let mut offsets: [u64; 4] = [0; 4];
        for i in 1..4 {
            offsets[i] = offsets[i-1] + indexer.size(i as u32 - 1);
        }
        return Indexes {
            indexer: indexer,
            offsets: offsets
        };
    }

    pub fn get_index(&self, cards: Vec<u8>, round: usize) -> u64 {
        return self.indexer.get_index(cards, round) + self.offsets[round];
    }

    pub fn get_hand(&self, round: u32, index: hand_index_t) -> Vec<u8> {
        return self.indexer.get_hand(round, index - self.offsets[round as usize]);
    }
}

fn main () {

    let indexes = Indexes::new();

    let mut file = File::open("ehs.dat").unwrap();

    let mut hand_mask = get_card_mask("2h3c".to_string());
    let mut board_mask = get_card_mask("KsQhJhTh9c".to_string());
    let mut cards: Vec<u8> = Vec::new();

    while hand_mask != 0 {
         cards.push(hand_mask.trailing_zeros() as u8);
         hand_mask = hand_mask & !(1u64 << hand_mask.trailing_zeros());
    }
    while board_mask != 0 {
         cards.push(board_mask.trailing_zeros() as u8);
         board_mask = board_mask & !(1u64 << board_mask.trailing_zeros());
    }
    // for c in cards.iter() {
    //     print!("{}{} ", RANKS[(c >> 2) as usize], SUITS[(c & 3) as usize]);
    // }
    // println!("");

    let test = [
        hand_indexer_t::init(1, [ 2 ].to_vec()),
        hand_indexer_t::init(2, [ 2, 3 ].to_vec()),
        hand_indexer_t::init(2, [ 2, 4 ].to_vec()),
        hand_indexer_t::init(2, [ 2, 5 ].to_vec()),
    ];
    let mut offsets = [0u64; 4];
    for i in 1..4 {
        offsets[i] = offsets[i-1] + test[i-1].size(if i == 1 { 0 } else { 1 });
    }

    // for i in 1..4 {
    //     println!("test {}", offsets[i]);
    //     println!("offsets {}", indexes.offsets[i]);
    // }


    let round = 1;

    let index = indexes.get_index(cards.clone(), round as usize);
    let test_index = test[round].get_index_last(cards);

    println!("Index {}", index);
    println!("Test {}", test_index);

    // let hand = indexes.get_hand(round, index);
    // for h in &hand {
    //     println!("{} {}", h >> 2, h & 3);
    // }

    // file.seek(SeekFrom::Start((index) * 4));
    // let float: f32 = file.unpack().unwrap();
    // println!("{}", float);
}
