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
mod kmeans;

use std::io::Write;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam::atomic::AtomicCell;

use bytepack::{ LEPacker };
use std::fs::OpenOptions;

use rand::distributions::{Uniform};
use rand::{SeedableRng, thread_rng, Rng};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use rayon::prelude::*;

use rust_poker::hand_range::{HandRange, HoleCards};
use rust_poker::equity_calculator::calc_equity;
use rust_poker::hand_indexer_s;

use kmeans::Kmeans;

use ehs::EHS;

const N_THREADS: usize = 8;

pub type Histogram = Vec<f32>;

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
fn get_bin(value: f32, bins: usize) -> usize {
    let interval = 1f32 / bins as f32;
    let mut bin = bins - 1;
    let mut threshold = 1f32 - interval;
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

    let samples_f = samples as f32;

    let card_dist: Uniform<u8> = Uniform::from(0..52);

    let cards_per_round = [2, 5, 6, 7];
    let round_size = ehs_table.indexers[round]
        .size(if round > 0 { 1 } else { 0 }) as usize;

    // number of hands to eval per thread
    let size_per_thread = (round_size / N_THREADS) as usize;

    // histograms to return
    let mut dataset = vec![vec![0f32; bins]; round_size];

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
                        print!("{:.3}% \r", (100 * j) as f32
                            / size_per_thread as f32);
                        io::stdout().flush().unwrap();
                    }

                    let index = ((i * size_per_thread) + j) as u64;
                    // get hand
                    ehs_table.indexers[round]
                        .get_hand(if round == 0 { 0 } else { 1}, index, cards.as_mut_slice());
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
                        let ehs = ehs_table.get_ehs(cards.as_slice()).unwrap() as f32;

                        slice[j][get_bin(ehs, bins)] += 1f32;
                    }
                    // normalize histogram
                    for k in 0..bins {
                        slice[j][k] /= samples_f;
                    }
                }
            });
        }
    }).unwrap();

    let duration = start_time.elapsed().as_millis();
    println!("Done.  Took {}ms", duration);

    return dataset;
}

fn generate_opponent_clusters(n_opp_clusters: usize) -> Vec<String> {

    let mut thread_rng = thread_rng();
    let n_samples = 10000usize;
    let n_bins = 100usize;
    let n_restarts: usize = 500;
    let ehs_table = EHS::new();

    let mut opp_features = generate_histograms(n_samples, 0, n_bins);
    let mut cards: Vec<u8> = vec![0; 2];
    let mut opp_ranges: Vec<(String, f32)> =
        vec![("".to_string(), 0f32); n_opp_clusters];

    let mut estimator = kmeans::Kmeans::init_random(
        n_restarts, n_opp_clusters,
        &mut thread_rng, &emd::emd_1d, &opp_features);

    println!("Running Kmeans");

    let opp_clusters = estimator.fit(&mut opp_features, &emd::emd_1d);

    // transform clusters into range string representation
    for i in 0..169 {
        ehs_table.indexers[0].get_hand(0, i as u64, cards.as_mut_slice());
        let hand_str = HoleCards(cards[0], cards[1]).to_string();
        let char_vec: Vec<char> = hand_str.chars().collect();
        if char_vec[0] == char_vec[2] {
            opp_ranges[opp_clusters[i]].0.push_str(&format!("{}{},", char_vec[0], char_vec[2]));
        } else if char_vec[1] == char_vec[3] {
            opp_ranges[opp_clusters[i]].0.push_str(&format!("{}{}s,", char_vec[0], char_vec[2]));
        } else {
            opp_ranges[opp_clusters[i]].0.push_str(&format!("{}{}o,", char_vec[0], char_vec[2]));
        }
    }

    // get all in equity for each hand range
    for i in 0..n_opp_clusters {
        // remove trailing comma
        opp_ranges[i].0.pop();
        // println!("{}", opp_ranges[i].0);
        let ranges = HandRange::from_strings([
                opp_ranges[i].0.to_string(),
                "random".to_string()
        ].to_vec());
        opp_ranges[i].1 = calc_equity(&ranges, 0, 1, 10000)[0] as f32;
    }

    // sort by all in equity
    opp_ranges.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // return sorted range string
    return opp_ranges.iter().map(|(s, _)| s.clone()).collect();
}

fn gen_ochs(round: u8) {
    let mut rng = thread_rng();
    let n_opp_clusters = 8;
    let ehs_table = EHS::new();
    let round_size = ehs_table.indexers[usize::from(round)].size(1);
    let total_cards: usize = match round {
        0 => 2,
        1 => 5,
        2 => 6,
        3 => 7,
        _ => panic!("invalid round")
    };


    println!("Generating OCHS ranges");
    let ohcs_ranges = generate_opponent_clusters(n_opp_clusters);
    for i in 0..ohcs_ranges.len() {
        println!("{}", ohcs_ranges[i]);
    }

    println!("Generating {} histograms for round {}", round_size, round);
    let mut histograms = vec![vec![0f32; 8]; round_size as usize];
    let acc = AtomicCell::new(0usize);
    histograms.par_iter_mut().enumerate().for_each(|(i, hist)| {

        let iteration = acc.fetch_add(1);
        if iteration % 1000 == 0 {
            print!("iteration {}/{}\r", iteration, round_size);
            io::stdout().flush().unwrap();
        }

        let mut cards = vec![0u8; total_cards];
        ehs_table.indexers[usize::from(round)]
            .get_hand(if round == 0 { 0 } else { 1 }, i as u64, cards.as_mut_slice());
        let hand_str = HoleCards(cards[0], cards[1]).to_string();

        let mut norm_sum = 0f32;
        for i in 0..n_opp_clusters {
            let hand_ranges = HandRange::from_strings([
                hand_str.to_owned(),
                ohcs_ranges[i].to_owned()
            ].to_vec());
            let mut board_mask = 0u64;
            for i in 2..total_cards {
                board_mask |= 1u64 << cards[i];
            }
            let e = calc_equity(&hand_ranges, board_mask, 1, 1000)[0] as f32;
            hist[i] = e;
            norm_sum += e;
        }
        for i in 0..n_opp_clusters {
            hist[i] /= norm_sum;
        }
    });


    // use 20% of data for testing
    let train_data = histograms
        .choose_multiple(&mut rng, round_size as usize / 5)
        .cloned()
        .collect();

    let n_restarts = 50;
    let n_clusters = 5000;
    let mut clusters: Vec<usize> = vec![0; histograms.len()];

    // initialize kmeans
    let mut estimator = kmeans::Kmeans::init_random(
        n_restarts, n_clusters,
        &mut rng, &kmeans::l2_dist, &train_data);

    // train kmeans
    estimator.fit(&train_data, &kmeans::l2_dist);
    estimator.predict(&histograms, &mut clusters, &kmeans::l2_dist);

    let mut file = OpenOptions::new().write(true).create_new(true).open("round_{}_ochs.dat").unwrap();
    for i in 0..round_size {
        file.pack(clusters[i as usize] as u32).unwrap();
    }
}

fn gen_emd(round: u8) {
    let mut rng = thread_rng();

    let hand_indexer = match round {
        0 => hand_indexer_s::init(1, vec![2]),
        1 => hand_indexer_s::init(2, vec![2, 3]),
        2 => hand_indexer_s::init(2, vec![2, 4]),
        3 => hand_indexer_s::init(2, vec![2, 5]),
        _ => panic!("Invalid round!")
    };

    let n_samples = 1000usize;
    let n_bins = 30usize;
    let n_clusters = 5000usize;
    let n_restarts: usize = 100;
    let round_size = hand_indexer.size(if round == 0 { 0 } else { 1 });

    let features = generate_histograms(n_samples, round.into(), n_bins);

    // use 30% of the data for training
    let train_data: Vec<Histogram> = features
        .choose_multiple(&mut rng, round_size / 3)
        .cloned()
        .collect();

    println!("Train data: {} {} {}", train_data[0].len(), train_data[1].len(), train_data[2].len());

    let mut clusters = vec![0usize; round_size];

    let mut estimator = kmeans::Kmeans::init_random(
        n_restarts, n_clusters,
        &mut rng, &emd::emd_1d, &train_data);

    estimator.fit(&train_data, &emd::emd_1d);

    estimator.predict(&features, &mut clusters, &emd::emd_1d);

    let mut file = OpenOptions::new().write(true).create_new(true).open(format!("round_{}_emd.dat", round)).unwrap();
    for i in 0..round_size {
        file.pack(clusters[i as usize] as u32).unwrap();
    }
}

fn main() {
    gen_emd(1); // flop
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
