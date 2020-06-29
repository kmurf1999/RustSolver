extern crate test;

use bytepack::{ LEUnpacker };
use std::io::prelude::*;
use std::fs::File;
use std::io::SeekFrom;

use rust_solver::hand_index::{ hand_indexer_t };

/**
 * structur to interface with EHS.dat table
 */
pub struct EHS {
    pub indexers: [hand_indexer_t; 4],
    // offsets for lookup table
    offsets: [u64; 4],
    // file pointer to lookup table
    // file_location: str
}

impl EHS {
    /**
     * create indexers and generate offsets
     */
    pub fn new() -> EHS {
        let indexers = [
            hand_indexer_t::init(1, [ 2 ].to_vec()),
            hand_indexer_t::init(2, [ 2, 3 ].to_vec()),
            hand_indexer_t::init(2, [ 2, 4 ].to_vec()),
            hand_indexer_t::init(2, [ 2, 5 ].to_vec()),
        ];
        let mut offsets: [u64; 4] = [0; 4];
        for i in 1..4 {
            offsets[i] = offsets[i-1]
                    + indexers[i-1].size(if i == 1 { 0 } else { 1 });
        }
        EHS {
            indexers: indexers,
            offsets: offsets
        }
    }

    /**
     * Get offset index of cards for lookup table
     * first two cards are hole cards
     */
    pub fn get_ehs(&self, cards: &[u8]) -> f32 {
        let mut file = File::open("ehs.dat").unwrap();
        let i: usize = match cards.len() {
            2 => 0,
            5 => 1,
            6 => 2,
            7 => 3,
            _ => 0
        };
        let index = self.indexers[i].get_index(cards);
        match file.seek(SeekFrom::Start((index + self.offsets[i]) * 4)) {
            Ok(_) => {
                let equity: f32 = file.unpack().unwrap();
                return equity;
            },
            Err(error) => {
                // TODO BETTER handling
                println!("{}", error);
                return 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use rand::distributions::{Uniform};
    use rand::{thread_rng, Rng};

    /**
     * return a vector of n random cards
     */
    fn random_cards(n_cards: usize) -> Vec<u8> {
        let mut cards: Vec<u8> = vec![0; n_cards];
        let card_dist: Uniform<u8> = Uniform::from(0..52);
        let mut card_mask: u64 = 0;
        let mut rng = thread_rng();
        for i in 0..n_cards {
            loop {
                cards[i] = rng.sample(card_dist);
                if (card_mask & (1u64 << cards[i])) == 0 {
                    card_mask |= 1u64 << cards[i];
                    break;
                }
            }
        }
        return cards;
    }

    #[bench]
    fn bench_get_ehs_flop(b: &mut Bencher) {
        let ehs_table = EHS::new();
        let cards = random_cards(5);
        b.iter(|| ehs_table.get_ehs(cards.as_slice()));
    }

    #[bench]
    fn bench_get_ehs_turn(b: &mut Bencher) {
        let ehs_table = EHS::new();
        let cards = random_cards(6);
        b.iter(|| ehs_table.get_ehs(cards.as_slice()));
    }

    #[bench]
    fn bench_get_ehs_river(b: &mut Bencher) {
        let ehs_table = EHS::new();
        let cards = random_cards(7);
        b.iter(|| ehs_table.get_ehs(cards.as_slice()));
    }

}
