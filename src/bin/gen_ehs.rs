extern crate rust_poker;
// extern crate rayon;
extern crate bytepack;
extern crate memmap;
extern crate crossbeam;

use std::io::Write; // <--- bring flush() into scope
use std::io;
use bytepack::{ LEPacker };
use std::time::{ Instant };
use std::fs::OpenOptions;

use rust_poker::equity_calc::EquityCalc;
use rust_poker::card_range::{ Combo, CardRange, RANKS, SUITS };
use rust_solver::hand_index::hand_indexer_t;

const N_THREADS: u64 = 8;

fn main() {
    let cards_per_round: [usize; 4] = [2, 5, 6, 7];

    // create preflop indexer
    let indexers = [
        hand_indexer_t::init(1, [ 2 ].to_vec()),
        hand_indexer_t::init(2, [ 2, 3 ].to_vec()),
        hand_indexer_t::init(2, [ 2, 4 ].to_vec()),
        hand_indexer_t::init(2, [ 2, 5 ].to_vec()),
    ];
    // let indexer = hand_indexer_t::init(4, [ 2, 3, 1, 1 ].to_vec());

    // let mut file = File::create("ehs.dat").unwrap();
    let mut file = OpenOptions::new().write(true).create_new(true).open("ehs.dat").unwrap();

    for i in 0..4 {
        let start_time = Instant::now();
        // number of isomorphic hands in this street
        let round = if i == 0 { 0 } else { 1 };
        let batch_size = indexers[i].size(round);
        println!("{} combinations in round {}", batch_size, i);
        // num hands per thread
        let size_per_thread = batch_size / N_THREADS;
        // equity table
        let mut equity_table = vec![0f32; batch_size as usize];
        // current round 0->preflop, 3->river
        crossbeam::scope(|scope| {
            for (j, slice) in equity_table.chunks_mut(size_per_thread as usize).enumerate() {
                scope.spawn(move || {
                    let mut board_mask: u64;
                    let mut combo: Combo;
                    let mut hand_ranges: Vec<CardRange>;
                    for k in 0..slice.len() {

                        // update percent every 1000 hands on thread 0
                        if (j == 0) && (k & 0xfff == 0) {
                            print!("{:.3}% \r", (100 * k) as f32 / size_per_thread as f32);
                            io::stdout().flush().unwrap();
                        }

                        let hand = indexers[i].get_hand(round, ((j as u64) * size_per_thread) + (k as u64));
                        combo = Combo(hand[0], hand[1]);

                        // create board
                        board_mask = 0;
                        let mut board_str = String::new();
                        for n in 2..cards_per_round[i as usize] {
                            board_mask |= 1u64 << hand[n];
                            board_str.push(RANKS[(hand[n] >> 2) as usize]);
                            board_str.push(SUITS[(hand[n] & 3) as usize]);
                        }

                        hand_ranges = CardRange::from_str_arr([combo.to_string(), "random".to_string()].to_vec());

                        // run sim
                        if i == 0 {
                            slice[k] = EquityCalc::start(&mut hand_ranges, board_mask, 1, 10000)[0];
                        } else { // small sample count and more cores
                            slice[k] = EquityCalc::start(&mut hand_ranges, board_mask, 2, 2000)[0];
                        }
                    }
                });
            }
        });

        // write to file
        file.pack_all(&equity_table[..]).unwrap();

        let duration = start_time.elapsed().as_millis();
        println!("round {} done. took {}ms ({:.2} iterations / ms)", i, duration, batch_size as f64 / duration as f64);
    }
}
