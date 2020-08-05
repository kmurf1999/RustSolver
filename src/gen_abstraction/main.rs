#![feature(test)]
#![allow(dead_code)]
#![allow(unused_imports)]

extern crate bytepack;
extern crate crossbeam;
extern crate rand;
extern crate rayon;
extern crate rust_poker;
extern crate test;

mod ehs;
mod emd;
mod kmeans;

use std::io;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam::atomic::AtomicCell;

use bytepack::LEPacker;
use std::fs::OpenOptions;

use rand::distributions::Uniform;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng, SeedableRng};

use rayon::prelude::*;

use rust_poker::equity_calculator::calc_equity;
use rust_poker::hand_indexer_s;
use rust_poker::hand_range::{char_to_rank, HandRange, HoleCards};

// use kmeans::Kmeans;

use ehs::EHS;

const N_THREADS: usize = 16;

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

    let start_time = Instant::now();

    let ehs_table = EHS::new();

    let samples_f = samples as f64;

    let card_dist: Uniform<u8> = Uniform::from(0..52);

    let cards_per_round = [2, 5, 6, 7];
    let round_size = ehs_table.indexers[round].size(if round > 0 { 1 } else { 0 }) as usize;

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
                        print!("{:.3}% \r", (100 * j) as f64 / size_per_thread as f64);
                        io::stdout().flush().unwrap();
                    }

                    let index = ((i * size_per_thread) + j) as u64;
                    // get hand
                    ehs_table.indexers[round].get_hand(
                        if round == 0 { 0 } else { 1 },
                        index,
                        cards.as_mut_slice(),
                    );
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
                        let ehs = ehs_table.get_ehs(cards.as_slice()).unwrap() as f64;

                        slice[j][get_bin(ehs, bins)] += 1f64;
                    }
                    // normalize histogram
                    for k in 0..bins {
                        slice[j][k] /= samples_f;
                    }
                }
            });
        }
    })
    .unwrap();

    let duration = start_time.elapsed().as_millis();
    println!("Done.  Took {}ms", duration);

    return dataset;
}

fn generate_opponent_clusters(n_opp_clusters: usize) -> Vec<String> {
    let mut thread_rng = thread_rng();
    let n_samples = 10000usize;
    let n_bins = 35usize;
    let ehs_table = EHS::new();

    let opp_features = generate_histograms(n_samples, 0, n_bins);
    let mut cards: Vec<u8> = vec![0; 2];
    let mut opp_ranges: Vec<(String, f64)> = vec![("".to_string(), 0f64); n_opp_clusters];

    // let mut estimator =
    // kmeans::Kmeans::init_pp(n_opp_clusters, &mut thread_rng, &emd::emd_1d, &opp_features);

    let mut estimator = kmeans::Kmeans::init_random(
        10,
        n_opp_clusters,
        &mut thread_rng,
        &emd::emd_1d,
        &opp_features,
    );
    // let mut estimator =
    // kmeans::Kmeans::init_pp(n_opp_clusters, &mut thread_rng, &emd::emd_1d, &opp_features);
    // println!("Running Kmeans");

    // estimator.growbatch_rho(&mut thread_rng, &emd::emd_1d, 10, &opp_features);
    // estimator.fit_regular(&opp_features, &emd::emd_1d);
    estimator.fit_growbatch(&mut thread_rng, &emd::emd_1d, 50, &opp_features);

    let mut opp_clusters = vec![0usize; opp_features.len()];
    let inertia = estimator.predict(&opp_features, &mut opp_clusters, &emd::emd_1d);
    println!("{}", inertia / n_opp_clusters as f64);

    // transform clusters into range string representation
    for i in 0..169 {
        ehs_table.indexers[0].get_hand(0, i as u64, cards.as_mut_slice());
        let hand_str = HoleCards(cards[0], cards[1]).to_string();
        let char_vec: Vec<char> = hand_str.chars().collect();
        let (c1, c2) = if char_to_rank(char_vec[0]) > char_to_rank(char_vec[2]) {
            (char_vec[0], char_vec[2])
        } else {
            (char_vec[2], char_vec[0])
        };
        if char_vec[0] == char_vec[2] {
            opp_ranges[opp_clusters[i]]
                .0
                .push_str(&format!("{}{},", c1, c2));
        } else if char_vec[1] == char_vec[3] {
            opp_ranges[opp_clusters[i]]
                .0
                .push_str(&format!("{}{}s,", c1, c2));
        } else {
            opp_ranges[opp_clusters[i]]
                .0
                .push_str(&format!("{}{}o,", c1, c2));
        }
    }

    // get all in equity for each hand range
    for i in 0..n_opp_clusters {
        // remove trailing comma
        opp_ranges[i].0.pop();
        // println!("{}", opp_ranges[i].0);
        let ranges =
            HandRange::from_strings([opp_ranges[i].0.to_string(), "random".to_string()].to_vec());
        opp_ranges[i].1 = calc_equity(&ranges, 0, 1, 10000)[0] as f64;
    }

    // sort by all in equity
    opp_ranges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // // return sorted range string
    return opp_ranges.iter().map(|(s, _)| s.clone()).collect();
}

// fn gen_ochs(round: u8, n_clusters: usize) {

//     if round < 1 || round > 3 {
//         panic!("Invalid round");
//     }

//     let mut rng = thread_rng();
//     let ehs_table = EHS::new();
//     let n_opp_clusters: usize = 8;
//     let round_size = ehs_table.indexers[usize::from(round)]
//         .size(1);
//     let total_cards: usize = match round {
//         0 => 2,
//         1 => 5,
//         2 => 6,
//         3 => 7,
//         _ => panic!("invalid round")
//     };

//     println!("Generating OCHS ranges");

//     let ohcs_ranges = generate_opponent_clusters(n_opp_clusters);
//     for i in 0..ohcs_ranges.len() {
//         println!("{}", ohcs_ranges[i]);
//     }

//     println!("Generating {} histograms for round {}", round_size, round);

//     let mut features = vec![vec![0f64; n_opp_clusters]; round_size as usize];
//     let acc = AtomicCell::new(0usize);
//     features.par_iter_mut().enumerate().for_each(|(i, hist)| {

//         let iteration = acc.fetch_add(1);
//         if iteration % 1000 == 0 {
//             print!("iteration {}/{}\r", iteration, round_size);
//             io::stdout().flush().unwrap();
//         }

//         let mut cards = vec![0u8; total_cards];
//         ehs_table.indexers[usize::from(round)]
//             .get_hand(if round == 0 { 0 } else { 1 }, i as u64, cards.as_mut_slice());
//         let hand_str = HoleCards(cards[0], cards[1]).to_string();

//         let mut norm_sum = 0f64;
//         for i in 0..n_opp_clusters {
//             let hand_ranges = HandRange::from_strings([
//                 hand_str.to_owned(),
//                 ohcs_ranges[i].to_owned()
//             ].to_vec());
//             let mut board_mask = 0u64;
//             for i in 2..total_cards {
//                 board_mask |= 1u64 << cards[i];
//             }
//             let e = calc_equity(&hand_ranges, board_mask, 1, 1000)[0] as f64;
//             hist[i] = e;
//             norm_sum += e;
//         }
//         for i in 0..n_opp_clusters {
//             hist[i] /= norm_sum;
//         }
//     });

//     // only use 20% of data for training
//     let train_data: Vec<Histogram> = features
//         .choose_multiple(&mut rng, round_size as usize / 5)
//         .cloned()
//         .collect();

//     let n_restarts = 50;
//     // let n_clusters = 5000;

//     let mut clusters: Vec<usize> = vec![0; round_size as usize];

//     // initialize kmeans
//     let mut estimator = kmeans::Kmeans::init_random(
//         n_restarts, n_clusters,
//         &mut rng, &kmeans::l2_dist, &train_data);

//     // train kmeans
//     estimator.fit_regular(&train_data, &kmeans::l2_dist);

//     estimator.predict(&features, &mut clusters, &kmeans::l2_dist);

//     println!("Clusters 0..5: {} {} {} {} {}",
//              clusters[0], clusters[1],
//              clusters[2], clusters[3],
//              clusters[4]);

//     let mut file = OpenOptions::new().write(true).create_new(true).open("round_{}_ochs.dat").unwrap();
//     for i in 0..round_size {
//         file.pack(clusters[i as usize] as u32).unwrap();
//     }
// }

/// generates emd abstraction and saves to file
///
/// # Arguments
///
/// * `round` round to make abstraction for 1: flop, 3: river
/// * `n_samples` number of samples per histogram
/// * `n_bins` number of bins in histogram
///
// fn gen_emd(round: u8, n_clusters: usize, n_samples: usize, n_bins: usize) {

//     if round < 1 || round > 3 {
//         panic!("invalid round");
//     }

//     let mut rng = thread_rng();

//     let hand_indexer = match round {
//         0 => hand_indexer_s::init(1, vec![2]),
//         1 => hand_indexer_s::init(2, vec![2, 3]),
//         2 => hand_indexer_s::init(2, vec![2, 4]),
//         3 => hand_indexer_s::init(2, vec![2, 5]),
//         _ => panic!("Invalid round!")
//     };

//     // let n_samples = 2000usize;
//     // let n_bins = 30usize;
//     // let n_clusters = 5000usize;
//     let n_restarts: usize = 25;
//     let round_size = hand_indexer.size(if round == 0 { 0 } else { 1 });

//     let features = generate_histograms(n_samples, round.into(), n_bins);
//     let mut clusters = vec![0usize; round_size as usize];

//     // let mut estimator = kmeans::Kmeans::init_pp(
//     //     n_clusters, &mut rng,
//     //     &emd::emd_1d, &features);

//     let mut estimator = kmeans::Kmeans::init_random(
//         n_restarts, n_clusters,
//         &mut rng, &emd::emd_1d, &features);

//     // use mini batches
//     // let (clusters, inertia) = estimator.fit_regular(&features, &emd::emd_1d);
//     estimator.fit_minibatch(&mut rng, &features, 100, 100_000, &emd::emd_1d);

//     estimator.predict(&features, &mut clusters, &emd::emd_1d);

//     let mut file = OpenOptions::new()
//         .write(true)
//         .create_new(true)
//         .open(format!("means_{}_round_{}_emd.dat", n_clusters, round))
//         .unwrap();

//     for i in 0..round_size {
//         file.pack(clusters[i as usize] as u32).unwrap();
//     }
// }

fn main() {
    // round, n means, n samples, 40 bins
    // gen_emd(1, 5000, 2500, 40); // flop
    // gen_emd(2, 5000, 2500, 30); // turn
    // round, n means
    // gen_ochs(3, 5000); // river
    let ranges = generate_opponent_clusters(8);
    for r in ranges {
        println!("{}", r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    // #[bench]
    // fn bench_gen_round_0(b: &mut Bencher) {
    //     b.iter(|| generate_round(0));
    // }
}
