#![feature(test)]
#![allow(dead_code)]
#![allow(unused_imports)]


extern crate test;
extern crate bytepack;
extern crate rand;
extern crate rust_poker;
extern crate crossbeam;
extern crate rayon;


mod ehs;
mod emd;
mod hand_index;
mod kmeans;

use std::io::Write; // <--- ring flush() into scope
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::distributions::{Uniform};
use rand::{SeedableRng, thread_rng, Rng};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use rayon::prelude::*;


use rust_poker::card_range::{CardRange, Combo};
use rust_poker::equity_calc::EquityCalc;

use kmeans::Kmeans;

use ehs::EHS;

const N_THREADS: usize = 8;

pub type Histogram = Vec<f64>;

/**
 * Create histograms for each combo
 *
 * For each round, get the hand from the index.
 * Then, randomly assign new turn and river cards
 * Evaluate the resulting hand and push probability to histogram
 */

/**
 * Get index of bin in histogram
 * @param bins: number of bins in histogram
 * @param value: the probability
 */
fn get_bin(value: f64, bins: usize) -> usize {
    let interval = 1f64 / bins as f64;
    let mut bin = bins - 1;
    let mut threshold = 1f64 - interval;
    while bin > 0 {
        if value > threshold {
            return bin;
        }
        bin -= 1;
        threshold -= interval;
    }
    return 0;
}

/**
 * Generates histograms based on EHS vs random probability distributions
 *
 * samples: number of samples per histogram
 * round: betting round (0 -> preflop, 3 -> river)
 * bins: number of bins per histogram
 */
fn generate_histograms(samples: usize, round: usize, bins: usize) -> Vec<Histogram> {

    let mut thread_rng = thread_rng();
    let ehs_table = EHS::new();
    let samples_f = samples as f64;

    let card_dist: Uniform<u8> = Uniform::from(0..52);

    let cards_per_round = [2, 5, 6, 7];
    let round_size = ehs_table.indexers[round]
        .size(if round > 0 { 1 } else { 0 }) as usize;

    // number of hands to eval per thread
    let size_per_thread = (round_size / N_THREADS) as usize;

    // histograms to return
    let mut dataset = vec![vec![0f64; bins]; round_size];

    println!("Generating {} histograms for round {}", round_size, round);

    crossbeam::scope(|scope| {
        for (i, slice) in dataset.chunks_mut(size_per_thread).enumerate() {
            // let ehs_table = Arc::clone(&ehs_table);
            let ehs_table = EHS::new();
            // let mut rng = SmallRng::from_entropy();
            let mut rng = SmallRng::from_rng(&mut thread_rng).unwrap();
            let mut cards: Vec<u8> = vec![0; 7];
            scope.spawn(move |_| {
                for j in 0..slice.len() {

                    if (i == 0) && (j & 0xff == 0) {
                        print!("{:.3}% \r", (100 * j) as f64
                            / size_per_thread as f64);
                        io::stdout().flush().unwrap();
                    }

                    let index = ((i * size_per_thread) + j) as u64;
                    // get hand
                    ehs_table.indexers[round]
                        .get_hand(round as u32, index, cards.as_mut_slice());
                    // build mask for rejection sampling
                    let mut card_mask: u64 = 0;
                    for k in 0..cards_per_round[round] {
                        card_mask |= 1u64 << cards[k];
                    }
                    // create histogram for index (i * size) + j
                    for _ in 0..samples {
                        // fill remaining board cards
                        let mut c_mask = card_mask;
                        for k in cards_per_round[round]..7 {
                            loop {
                                cards[k] = rng.sample(card_dist);
                                if (c_mask & 1u64 << cards[k]) == 0 {
                                    c_mask |= 1u64 << cards[k];
                                    break;
                                }
                            }
                        }
                        // get ehs and add to histogram
                        let ehs = ehs_table.get_ehs(cards.as_slice()).unwrap();

                        slice[j][get_bin(ehs, bins)] += 1f64;
                    }
                    // normalize histogram
                    for k in 0..bins {
                        slice[j][k] /= samples_f;
                    }
                }
            });
        }
    }).unwrap();

    return dataset;
}

// fn generate_opponent_clusters<R: Rng>(n_opp_clusters: usize, mut rng: &mut R) -> Vec<String> {
// 
//     let samples = 1000usize;
//     let bins = 30usize;
//     let restarts: usize = 10;
//     let ehs_table = EHS::new();
// 
//     let mut opp_features = generate_histograms(samples, 0, bins);
//     let mut opp_clusters: Vec<Histogram> = Vec::with_capacity(n_opp_clusters);
//     let mut cards: Vec<u8> = vec![0; 2];
//     let mut opp_ranges: Vec<(String, f64)> = vec![("".to_string(), 0f64); n_opp_clusters];
// 
//     println!("Initializing opponent centers using Kmeans++ with {} restarts", restarts);
//     kmeans::kmeans_center_multiple_restarts(
//             restarts, n_opp_clusters,
//             &mut opp_clusters, &opp_features,
//             &emd::emd_1d, &mut rng);
// 
//     println!("Running Kmeans");
//     kmeans::kmeans(n_opp_clusters, &mut opp_features,
//             &emd::emd_1d, &mut opp_clusters);
// 
// 
//     // transform clusters into range string representation
//     for i in 0..opp_features.len() {
//         ehs_table.indexers[0].get_hand(0, i as u64, cards.as_mut_slice());
//         let hand_str = Combo(cards[0], cards[1]).to_string();
//         opp_ranges[opp_features[i].cluster].0.push_str(&hand_str);
//         opp_ranges[opp_features[i].cluster].0.push_str(",");
//     }
// 
//     // get all in equity for each hand range
//     for i in 0..n_opp_clusters {
//         opp_ranges[i].0.pop(); // remove trailing comma
//         let mut hand_range = CardRange::from_str_arr([
//                 opp_ranges[i].0.to_string(),
//                 "random".to_string()
//         ].to_vec());
//         let e = EquityCalc::start(&mut hand_range, 0, 1, 10000)[0];
//         opp_ranges[i].1 = e;
//     }
//     // sort by all in equity
//     opp_ranges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
// 
//     // return sorted range string
//     return opp_ranges.iter().map(|(s, _)| s.clone()).collect();
// }

// fn generate_ohcs(round: usize, cluster: usize) {
//     let mut thread_rng = thread_rng();
//     let mut rng = SmallRng::from_rng(&mut thread_rng).unwrap();
// 
//     let n_opp_clusters = 8usize;
//     let opp_clusters = Arc::new(generate_opponent_clusters(n_opp_clusters, &mut rng));
//     let cards_per_round = vec![2usize, 5, 6, 7];
//     let cards_this_round = cards_per_round[round];
// 
//     // // to get hand
//     let ehs_table = EHS::new();
// 
//     // // Generate histograms for each hand in round
//     let round_size = ehs_table.indexers[round]
//         .size(if round > 0 { 1 } else { 0 }) as usize;
// 
//     let size_per_thread = (round_size / N_THREADS) as usize;
// 
//     println!("Generating {} histograms for round {}", round_size, round);
// 
//     let mut features: Arc<Mutex<Vec<DataPoint>>> =
//         Arc::new(Mutex::new(Vec::with_capacity(round_size)));
// 
//     let mut acc = 0usize;
// 
//     crossbeam::scope(|scope| {
//         for i in 0..N_THREADS {
// 
//             let start = acc;
//             acc += size_per_thread;
// 
//             // make last thread bigger to compensate
//             if i == N_THREADS - 1 {
//                 acc = round_size;
//             }
// 
//             let f_lock = features.clone();
//             let opp_clusters = opp_clusters.clone();
//             scope.spawn(move |_| {
//                 let mut cards = vec![0u8; cards_this_round];
//                 let mut board_mask: u64 = 0;
//                 let mut hand_range: Vec<CardRange> = Vec::with_capacity(2);
//                 let ehs_table = EHS::new();
//                 for j in start..acc {
//                     if i == 0 && (j & 0xff) == 0 {
//                         print!("{:.3}% \r", (100 * j) as f64 / (acc - start) as f64);
//                         io::stdout().flush().unwrap();
//                     }
// 
//                     // create histogram feature
//                     let mut feature = DataPoint {
//                         hand_index: j,
//                         cluster: 0,
//                         histogram: vec![0f64; n_opp_clusters]
//                     };
// 
//                     ehs_table.indexers[round]
//                         .get_hand(if round == 0 { 0 } else { 1 }, j as u64, cards.as_mut_slice());
// 
//                     let hand = Combo(cards[0], cards[1]).to_string();
// 
//                     // get board mask
//                     board_mask = 0;
//                     for k in 2..cards_this_round {
//                         board_mask |= 1u64 << cards[k];
//                     }
// 
// 
//                     let mut sum = 0f64;
//                     for k in 0..n_opp_clusters {
//                         hand_range = CardRange::from_str_arr([
//                             hand.clone(),
//                             opp_clusters[k].clone()
//                         ].to_vec());
//                         // println!("{} {:#066b}", hand.clone(), board_mask);
//                         let e = EquityCalc::start(
//                             &mut hand_range, board_mask,
//                             1, 1000
//                         )[0];
//                         feature.histogram[k] = e;
//                         sum += e;
//                     }
//                     // normalize
//                     for k in 0..n_opp_clusters {
//                         feature.histogram[k] /= sum;
//                     }
// 
//                     f_lock.lock().unwrap().push(feature);
//                 }
//             });
//         }
//     }).unwrap();

    // let mut features = box[DataPoint {
    //     cluster: 0,
    //     histogram: box[0f64; n_opp_clusters]
    // }; round_size];
    

    // crossbeam::scope(|scope| {
    //     let opp_ranges = Arc::new(opp_ranges);
    //     for (i, slice) in features.chunks_mut(size_per_thread).enumerate() {
    //         let ehs_table = EHS::new();
    //         // let mut rng = SmallRng::from_rng(&mut thread_rng).unwrap();
    //         let mut cards: Vec<u8> = vec![0; cards_this_round];
    //         let opp_ranges = Arc::clone(&opp_ranges);
    //         scope.spawn(move |_| {
    //             for j in 0..slice.len() {
    //                 // update progress to stdout
    //                 if (i == 0) {
    //                     print!("{:.3}% \r", (100 * j) as f64
    //                         / size_per_thread as f64);
    //                     io::stdout().flush().unwrap();
    //                 }
    //                 let index = ((i * size_per_thread) + j) as u64;

    //                 ehs_table.indexers[round]
    //                     .get_hand(round as u32, index, cards.as_mut_slice());

    //                 // get board mask
    //                 let mut board_mask: u64 = 0;
    //                 for k in 2..cards_this_round {
    //                     board_mask |= 1u64 << cards[k];
    //                 }

    //                 let mut sum: f64 = 0.0;
    //                 for k in 0..n_opp_clusters {
    //                     let mut hand_range = CardRange::from_str_arr([
    //                         Combo(cards[0], cards[1]).to_string(),
    //                         opp_ranges[k].0.clone()
    //                     ].to_vec());
    //                     let e = EquityCalc::start(
    //                             &mut hand_range, board_mask,
    //                             1, 10000)[0];
    //                     slice[j].histogram[k] = e;
    //                     sum += e;
    //                 }
    //                 // normalize
    //                 for k in 0..n_opp_clusters {
    //                     slice[j].histogram[k] /= sum;
    //                 }

    //             }
    //         });
    //     }
    // }).unwrap();
// }

fn generate_round(round: usize) {
    let mut thread_rng = thread_rng();

    // to grab expected hand strengths
    let ehs_table = EHS::new();

    // generate histograms
    let bins: usize = 30;
    let samples: usize = 100;
    let dataset = generate_histograms(samples, round, bins);

    let n_restarts: usize = 50;
    let n_centers: usize = 10;

    let mut rng = SmallRng::from_rng(&mut thread_rng).unwrap();

    // init kmeans clusters
    let mut estimator = Kmeans::init_random(n_restarts,
            n_centers, &mut rng,
            &emd::emd_1d, &dataset);

    // fit to train data (20% of full dataset)
    let batch_size = dataset.len();
    let batches: usize = 1;
    let start = Instant::now();
    // for i in 0..batches {
        // let train_data = dataset.choose_multiple(&mut rng, batch_size).cloned().collect();
        estimator.fit(&dataset, &emd::emd_1d);
    // }

    let elapsed = start.elapsed().as_millis();

    // get clusters for full dataset
    let mut clusters = vec![0usize; dataset.len()];
    estimator.predict(&dataset, &mut clusters, &emd::emd_1d);

    let mut cards: Vec<u8> = vec![0; 2];
    for i in 0usize..169 {
        ehs_table.indexers[0].get_hand(0, i as u64, cards.as_mut_slice());
        let cards = Combo(cards[0], cards[1]).to_string();
        println!("{} bin {}", cards, clusters[i]);
    }

    println!("Took {}ms", elapsed);
}

fn main () {
    generate_round(0);
    // generate_ohcs(3, 100);
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_gen_round_0(b: &mut Bencher) {
        b.iter(|| generate_round(0));
    }
}
