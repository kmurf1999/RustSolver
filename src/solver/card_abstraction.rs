use hashbrown::HashMap;
use rust_poker::hand_indexer_s;
use rust_poker::hand_range::HandRange;
use rust_poker::equity_calculator::remove_invalid_combos;
use crate::state::BettingRound;
use rayon::prelude::*;
use std::sync::mpsc::channel;
use std::thread;

/// Card abstraction interface for a single round
pub struct ISOMORPHIC {
    hand_indexer: hand_indexer_s,
    /// hand_index -> cluster_idx for each player
    cluster_map: Vec<HashMap<u64,usize>>,
    /// the number of clusters for each player
    size: Vec<usize>
}

pub struct EMD {

}

pub struct OCHS {

}

pub trait CardAbstraction {
    type AbsType;
    fn init(hand_ranges: &Vec<HandRange>, board_mask: u64, round: BettingRound) -> Self::AbsType;
    fn get_cluster(&self, cards: &[u8], player: u8) -> usize;
    fn get_size(&self, player: u8) -> usize;
}

impl CardAbstraction for ISOMORPHIC {

    type AbsType = ISOMORPHIC;

    fn init(hand_ranges: &Vec<HandRange>, initial_board_mask: u64, round: BettingRound) -> Self::AbsType {

        const N_THREADS: usize = 4;
        const CHANNEL_SIZE: usize = 10;

        let n_players = hand_ranges.len();
        let n_board_cards = initial_board_mask.count_ones() as usize;

        let hand_indexer = match round {
            BettingRound::Flop => hand_indexer_s::init(2, vec![2, 3]),
            BettingRound::Turn => hand_indexer_s::init(2, vec![2, 4]),
            BettingRound::River => hand_indexer_s::init(2, vec![2, 5])
        };

        let mut cards = [0u8; 7];
        let mut board_mask = initial_board_mask;
        for i in 0..n_board_cards {
            cards[i+2] = board_mask.trailing_zeros() as u8;
            board_mask ^= 1u64 << board_mask.trailing_zeros();
        }

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

                let (tx, rx) = crossbeam::channel::bounded::<u64>(1000);

                let consumer = scope.spawn(move |_| {
                    for r in rx {
                        if !map[0].contains_key(&r) {
                            map[0].insert(r, s[0]);
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
                            tx.send(index).unwrap();
                        },
                        1 => {
                            let used_card_mask =
                                hand_mask | initial_board_mask;
                            for i in 0u8..52 {
                                if ((1u64 << i) & used_card_mask) != 0 {
                                    continue;
                                }
                                cards[5] = i;

                                let index = hand_indexer.get_index(&cards);
                                tx.send(index).unwrap();
                            }
                        },
                        2 => {
                            let used_card_mask =
                                hand_mask | initial_board_mask;
                            for i in 0u8..52 {
                                if ((1u64 << i) & used_card_mask) != 0 {
                                    continue;
                                }
                                cards[5] = i;
                                let used_card_mask =
                                    (1u64 << i) | used_card_mask;
                                for j in 0..i {
                                    if ((1u64 << j) & used_card_mask) != 0 {
                                        continue;
                                    }
                                    cards[6] = j;
                                    let index = hand_indexer.get_index(&cards);
                                    tx.send(index).unwrap();
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

        ISOMORPHIC {
            hand_indexer: hand_indexer,
            cluster_map,
            size
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
#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use crate::options;
    use std::time::{Duration, Instant};

    #[bench]
    fn bench_init_turn(b: &mut Bencher) {
        // 4,418,307 ns/iter (+/- 152,791)
        let round = BettingRound::Turn;
        let flop_mask: u64 = 0b111;
        let mut hand_ranges = HandRange::from_strings(["random".to_string(), "random".to_string()].to_vec());
        remove_invalid_combos(&mut hand_ranges, flop_mask);
        b.iter(|| {
            let card_abs = ISOMORPHIC::init(&hand_ranges, flop_mask, round);
            assert_eq!(card_abs.size[1], 12888);
        });
    }

    #[test]
    fn test_init_river() {
        let round = BettingRound::River;
        let flop_mask: u64 = 0b11111;
        let mut hand_ranges = HandRange::from_strings(["random".to_string(), "random".to_string()].to_vec());
        remove_invalid_combos(&mut hand_ranges, flop_mask);
        let card_abs = ISOMORPHIC::init(&hand_ranges, flop_mask, round);
        assert_eq!(331, card_abs.size[0]);
        assert_eq!(331, card_abs.size[1]);
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
}
