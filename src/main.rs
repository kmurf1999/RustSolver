#![feature(test)]

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

use std::sync::Arc;
use rand::distributions::{Uniform};
use rand::{SeedableRng, thread_rng, Rng};
use rand::rngs::SmallRng;

use ehs::EHS;

const N_THREADS: usize = 8;

pub type Histogram = Vec<f32>;

#[derive(Debug, Clone)]
pub struct DataPoint {
    cluster: usize, // which bucket it belongs to
    histogram: Histogram // hisogram
}

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

fn generate_round(round: usize) {
    // CONSTANTS
    let mut thread_rng = thread_rng();
    let card_dist: Uniform<u8> = Uniform::from(0..52);

    // number of samples per histogram
    let samples_per_round = [1000, 1000, 1000];

    // to grab expected hand strengths
    let ehs_table = Arc::new(EHS::new());
    // number of bins in histogram
    let bins = 30;
    let cards_per_round = [2, 5, 6, 7];

    // NON CONSTS
    // number of hands in round
    let round_size = ehs_table.indexers[round].size(if round > 0 { 1 } else { 0 }) as usize;

    // numer of hands to eval per thread
    let size_per_thread = (round_size / N_THREADS) as usize;

    let mut features = vec![DataPoint {
        cluster: 0,
        histogram: vec![0f32; bins]
    }; round_size];
    // println!("Generating {} histograms for round {}", round_size, round);

    crossbeam::scope(|scope| {
        for (i, slice) in features.chunks_mut(size_per_thread).enumerate() {
            let ehs_table = Arc::clone(&ehs_table);
            // let mut rng = SmallRng::from_entropy();
            let mut rng = SmallRng::from_rng(&mut thread_rng).unwrap();
            let mut cards: Vec<u8> = vec![0; 7];
            scope.spawn(move || {
                for j in 0..slice.len() {
                    // print progress
                    // if (i == 0) && (j % 1000 == 0) {
                    //     print!("{:.3}% \r", (100 * j) as f32 / size_per_thread as f32);
                    //     io::stdout().flush().unwrap();
                    // }
                    let index = ((i * size_per_thread) + j) as u64;
                    // get hand
                    ehs_table.indexers[round].get_hand(round as u32, index, cards.as_mut_slice());
                    // build mask for rejection sampling
                    let mut card_mask: u64 = 0;
                    for k in 0..cards_per_round[round] {
                        card_mask |= 1u64 << cards[k];
                    }
                    // create histogram for index (i * size) + j
                    for _ in 0..samples_per_round[round] {
                        let mut card_mask = card_mask.clone();
                        // fill remaining board cards
                        for k in cards_per_round[round]..7 {
                            loop {
                                // cards[k] = card_dist.sample(&mut rng);
                                cards[k] = rng.sample(card_dist);
                                if (card_mask & 1u64 << cards[k]) == 0 {
                                    card_mask |= 1u64 << cards[k];
                                    break;
                                }
                            }
                        }
                        // get ehs and add to histogram
                        let ehs = ehs_table.get_ehs(cards.as_slice());
                        slice[j].histogram[get_bin(ehs, bins)] += 1f32;
                    }
                    // normalize histogram
                    for k in 0..bins {
                        slice[j].histogram[k] /= samples_per_round[round] as f32;
                    }
                }
            });
        }
    });

    let restarts: usize = 100;
    let n_buckets: usize = 300;
    let n_features: usize = features[0].histogram.len();
    let mut center: Vec<Histogram> = Vec::with_capacity(n_buckets);
    let mut rng = SmallRng::from_rng(&mut thread_rng).unwrap();

    kmeans::kmeans_center_multiple_restarts(
            restarts, n_buckets,
            &mut center, &features,
            &mut rng);
}


// fn kmeans(n_clusters: usize, dataset: &Vec<DataPoint>
//           center: &Vec<Histogram>, cost_matrix: &Vec<Vec<f32>>) {
// 
// }



fn main () {
    // generate_round(0);
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
