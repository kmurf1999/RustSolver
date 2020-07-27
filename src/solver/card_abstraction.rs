use hashbrown::HashMap;
use rust_poker::hand_indexer_s;
use rust_poker::hand_range::HandRange;
use rust_poker::equity_calculator::remove_invalid_combos;
use rayon::prelude::*;
use std::sync::mpsc::channel;
use std::thread;
use std::io::prelude::*;
use std::fs::File;
use std::io::SeekFrom;
use std::io::BufReader;
use combine::Parser;
use combine::parser::byte::num::le_u32;
use std::io::{Error, ErrorKind};
use bytepack::{LEUnpacker};


use crate::state::BettingRound;

fn index_to_cluster(index: u64, cluster_arr: Option<&Vec<u32>>) -> u64 {
    match cluster_arr {
        Some(arr) => {
            return arr[index as usize].into();
        },
        None => {
            return index;
        }
    }
}

/// Card abstraction interface for a single round
#[derive(Debug)]
pub struct ISOMORPHIC {
    hand_indexer: hand_indexer_s,
    /// hand_index -> cluster_idx for each player
    cluster_map: Vec<HashMap<u64,usize>>,
    /// the number of clusters for each player
    size: Vec<usize>
}

#[derive(Debug)]
pub struct EMD {
    hand_indexer: hand_indexer_s,
    /// hand_index -> cluster_idx for each player
    cluster_map: Vec<HashMap<u64,usize>>,
    /// the number of clusters for each player
    size: Vec<usize>,
    cluster_arr: Vec<u32>,
}

#[derive(Debug)]
pub struct OCHS {
    hand_indexer: hand_indexer_s,
    /// hand_index -> cluster_idx for each player
    cluster_map: Vec<HashMap<u64,usize>>,
    /// the number of clusters for each player
    size: Vec<usize>,
    cluster_arr: Vec<u32>,
}

#[derive(Debug)]
pub enum CardAbstraction {
    EMD(EMD),
    ISOMORPHIC(ISOMORPHIC),
    OCHS(OCHS)
}

pub trait ICardAbstraction {
    type AbsType;
    fn init(hand_ranges: &Vec<HandRange>, board_mask: u64, round: BettingRound) -> Self::AbsType;
    fn get_cluster(&self, cards: &[u8], player: u8) -> usize;
    fn get_size(&self, player: u8) -> usize;
}

fn generate_maps(
    hand_ranges: &Vec<HandRange>,
    initial_board_mask: u64,
    round: BettingRound,
    cluster_arr: Option<&Vec<u32>>)
    -> (Vec<usize>, Vec<HashMap<u64,usize>>, hand_indexer_s) {

    const CHANNEL_SIZE: usize = 10;

    let n_players = hand_ranges.len();
    let n_board_cards = initial_board_mask.count_ones() as usize;

    let (card_count, hand_indexer) = match round {
        BettingRound::Flop => (5, hand_indexer_s::init(2, vec![2, 3])),
        BettingRound::Turn => (6, hand_indexer_s::init(2, vec![2, 4])),
        BettingRound::River => (7, hand_indexer_s::init(2, vec![2, 5]))
    };

    let mut cards = vec![0u8; card_count];
    let mut board_mask = initial_board_mask;
    for i in 0..n_board_cards {
        cards[i+2] = board_mask.trailing_zeros() as u8;
        board_mask ^= 1u64 << board_mask.trailing_zeros();
    }

    let next_index = (initial_board_mask.count_ones() + 2) as usize;
    let cards_left = match round {
        BettingRound::Flop => 3 - n_board_cards,
        BettingRound::Turn => 4 - n_board_cards,
        BettingRound::River => 5 - n_board_cards
    };
    let mut cluster_map = vec![HashMap::new(); n_players];
    let mut size = vec![0usize; 2];

    crossbeam::scope(|scope| {
        let iter = cluster_map.chunks_mut(1).zip(size.chunks_mut(1)).into_iter();
        iter.enumerate().for_each(|(i, (map, s))| {

            let (tx, rx) = crossbeam::channel::bounded::<u64>(CHANNEL_SIZE);

            let consumer = scope.spawn(move |_| {
                for index in rx {
                    // get cluster index from file
                    if !map[0].contains_key(&index) {
                        map[0].insert(index, s[0]);
                        s[0] += 1;
                    }
                }
            });

            hand_ranges[i].hands.par_iter().for_each(|hand| {
                let tx = tx.clone();
                let hand_mask = (1u64 << hand.0) | (1u64 << hand.1);
                let mut cards = cards.clone();
                cards[0] = hand.0;
                cards[1] = hand.1;
                match cards_left {
                    0 => {
                        let index = hand_indexer.get_index(&cards);
                        let cluster = index_to_cluster(index, cluster_arr);
                        tx.send(cluster).unwrap();
                    },
                    1 => {

                        let used_card_mask =
                            hand_mask | initial_board_mask;
                        for i in 0u8..52 {
                            if ((1u64 << i) & used_card_mask) != 0 {
                                continue;
                            }
                            cards[next_index] = i;

                            let index = hand_indexer.get_index(&cards);
                            let cluster = index_to_cluster(index, cluster_arr);
                            tx.send(cluster).unwrap();
                        }
                    },
                    2 => {
                        let used_card_mask =
                            hand_mask | initial_board_mask;
                        for i in 0u8..52 {
                            if ((1u64 << i) & used_card_mask) != 0 {
                                continue;
                            }
                            cards[next_index] = i;
                            let used_card_mask =
                                (1u64 << i) | used_card_mask;
                            for j in 0..i {
                                if ((1u64 << j) & used_card_mask) != 0 {
                                    continue;
                                }
                                cards[next_index+1] = j;
                                let index = hand_indexer.get_index(&cards);
                                let cluster = index_to_cluster(index, cluster_arr);
                                tx.send(cluster).unwrap();
                            }
                        }
                    },
                    _ => panic!("invalid number of board cards")
                }
                drop(tx);
            });
            drop(tx);
            consumer.join().unwrap();
        });
    }).unwrap();


    return (size, cluster_map, hand_indexer);
}

impl ICardAbstraction for ISOMORPHIC {

    type AbsType = ISOMORPHIC;

    fn init(hand_ranges: &Vec<HandRange>, initial_board_mask: u64, round: BettingRound) -> Self::AbsType {
        let (size, cluster_map, hand_indexer) = generate_maps(
            hand_ranges,
            initial_board_mask,
            round,
            None);

        ISOMORPHIC {
            size,
            cluster_map,
            hand_indexer
        }
    }

    fn get_cluster(&self, cards: &[u8], player: u8) -> usize {
        let hand_index = self.hand_indexer.get_index(cards);
        return *self.cluster_map[usize::from(player)]
            .get(&hand_index)
            .unwrap();
    }

    fn get_size(&self, player: u8) -> usize {
        return self.size[usize::from(player)];
    }
}

impl ICardAbstraction for OCHS {
    type AbsType = OCHS;

    fn init(hand_ranges: &Vec<HandRange>, initial_board_mask: u64, round: BettingRound) -> Self::AbsType {

        let round_as_int = match round {
            BettingRound::Flop => 1,
            BettingRound::Turn => 2,
            BettingRound::River => 3,
        };

        let mut file = File::open(format!("round_{}_ochs.dat", round_as_int)).unwrap();
        let mut cluster_arr: Vec<u32> = Vec::new();
        file.unpack_to_end(&mut cluster_arr).unwrap();

        let (size, cluster_map, hand_indexer) = generate_maps(
            hand_ranges,
            initial_board_mask,
            round,
            Some(&cluster_arr));

        OCHS {
            size,
            cluster_map,
            cluster_arr,
            hand_indexer
        }
    }

    fn get_cluster(&self, cards: &[u8], player: u8) -> usize {
        let hand_index = self.hand_indexer.get_index(cards);
        let cluster_idx = self.cluster_arr[hand_index as usize];
        return *self.cluster_map[usize::from(player)]
            .get(&cluster_idx.into())
            .unwrap();
    }

    fn get_size(&self, player: u8) -> usize {
        return self.size[usize::from(player)];
    }
}

impl ICardAbstraction for EMD {
    type AbsType = EMD;

    fn init(hand_ranges: &Vec<HandRange>, initial_board_mask: u64, round: BettingRound) -> Self::AbsType {

        let round_as_int = match round {
            BettingRound::Flop => 1,
            BettingRound::Turn => 2,
            BettingRound::River => 3,
        };

        let mut file = File::open(format!("round_{}_emd.dat", round_as_int)).unwrap();
        let mut cluster_arr: Vec<u32> = Vec::new();
        file.unpack_to_end(&mut cluster_arr).unwrap();

        let (size, cluster_map, hand_indexer) = generate_maps(
            hand_ranges,
            initial_board_mask,
            round,
            Some(&cluster_arr));

        EMD {
            size,
            cluster_map,
            cluster_arr,
            hand_indexer
        }
    }

    fn get_cluster(&self, cards: &[u8], player: u8) -> usize {
        let hand_index = self.hand_indexer.get_index(cards);
        let cluster_idx = self.cluster_arr[hand_index as usize];
        return *self.cluster_map[usize::from(player)]
            .get(&cluster_idx.into())
            .unwrap();
    }

    fn get_size(&self, player: u8) -> usize {
        return self.size[usize::from(player)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use crate::options;
    use std::time::{Duration, Instant};

    #[test]
    fn test_init_iso_turn() {
        let round = BettingRound::Turn;
        let flop_mask: u64 = 0b111;
        let mut hand_ranges = HandRange::from_strings(["random".to_string(), "random".to_string()].to_vec());
        remove_invalid_combos(&mut hand_ranges, flop_mask);
        let card_abs = ISOMORPHIC::init(&hand_ranges, flop_mask, round);
        
        assert_eq!(12888, card_abs.size[0]);
        assert_eq!(12888, card_abs.size[1]);
        // test some indexes
        assert_eq!(
            card_abs.get_cluster(&[51u8, 5, 0, 1, 2, 3, 4], 0),
            card_abs.get_cluster(&[50u8, 5, 0, 1, 2, 3, 4], 0)
        );
        assert_eq!(
            card_abs.get_cluster(&[51u8, 5, 0, 1, 2, 3, 4], 1),
            card_abs.get_cluster(&[50u8, 5, 0, 1, 2, 3, 4], 1)
        );
        assert_ne!(
            card_abs.get_cluster(&[6, 5, 0, 1, 2, 3, 4], 1),
            card_abs.get_cluster(&[50u8, 5, 0, 1, 2, 3, 4], 1)
        );
    }

    #[test]
    fn test_init_emd_flop() {
        let round = BettingRound::Flop;
        let flop_mask: u64 = 0x2200100000000;
        let mut hand_ranges = HandRange::from_strings(["random".to_string(), "random".to_string()].to_vec());
        remove_invalid_combos(&mut hand_ranges, flop_mask);
        let card_abs = EMD::init(&hand_ranges, flop_mask, round);
        assert_eq!(107, card_abs.size[0]);
        assert_eq!(107, card_abs.size[1]);
    }
}
