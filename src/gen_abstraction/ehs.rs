// use bytepack::{ LEUnpacker };
use std::io::prelude::*;
use std::fs::File;
use std::io::SeekFrom;
use std::io::BufReader;
use combine::Parser;
use combine::parser::byte::num::le_f64;
use std::io::{Error, ErrorKind};

use hand_indexer::hand_index::hand_indexer_t;

/**
 * structur to interface with EHS.dat table
 */
pub struct EHS {
    pub indexers: [hand_indexer_t; 4],
    // offsets for lookup table
    offsets: [u64; 4],
    file: File
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
            offsets: offsets,
            file: File::open("ehs.dat").unwrap()
        }
    }

    /**
     * Get offset index of cards for lookup table
     * first two cards are hole cards
     */
    pub fn get_ehs(&self, cards: &[u8]) -> std::io::Result<f64> {
        let mut reader = BufReader::with_capacity(8, &self.file);
        //  let mut file = File::open("ehs.dat").unwrap();
        let i: usize = match cards.len() {
            2 => 0,
            5 => 1,
            6 => 2,
            7 => 3,
            _ => {
                println!("ERRROR");
                1000
            }
        };

        let index = self.indexers[i].get_index(cards);
        reader.seek(SeekFrom::Start((index + self.offsets[i]) * 8))?;
        let buffer = reader.fill_buf()?;
        let result = le_f64().parse(buffer);
        match result {
            Ok((val, _)) => {
                return Ok(val);
            },
            Err(_) => {
                return Err(Error::new(ErrorKind::Other, "Unexpected Parse"));
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

    #[test]
    fn test_get_ehs_aa() {
        let ehs_table = EHS::new();
        let mut cards: Vec<u8>;
        let mut ehs: f64;
        cards = vec![48u8, 49];
        ehs = ehs_table.get_ehs(cards.as_slice()).unwrap();
        assert_eq!(ehs, 0.8520068359375);
        cards = vec![48u8, 49, 0, 10, 20];
        ehs = ehs_table.get_ehs(cards.as_slice()).unwrap();
        assert_eq!(ehs, 0.865142822265625);
        cards = vec![48u8, 49, 0, 10, 20, 25];
        ehs = ehs_table.get_ehs(cards.as_slice()).unwrap();
        assert_eq!(ehs, 0.838623046875);
        cards = vec![48u8, 49, 0, 10, 20, 25, 29];
        ehs = ehs_table.get_ehs(cards.as_slice()).unwrap();
        assert_eq!(ehs, 0.842742919921875);
    }

    #[bench]
    fn bench_get_ehs_flop(b: &mut Bencher) {
        let ehs_table = EHS::new();
        let cards = random_cards(5);
        b.iter(|| ehs_table.get_ehs(cards.as_slice()).unwrap());
    }

    #[bench]
    fn bench_get_ehs_turn(b: &mut Bencher) {
        let ehs_table = EHS::new();
        let cards = random_cards(6);
        b.iter(|| ehs_table.get_ehs(cards.as_slice()).unwrap());
    }

    #[bench]
    fn bench_get_ehs_river(b: &mut Bencher) {
        let ehs_table = EHS::new();
        let cards = random_cards(7);
        b.iter(|| ehs_table.get_ehs(cards.as_slice()).unwrap())
    }

}
