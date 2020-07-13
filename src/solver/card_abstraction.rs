use hashbrown::HashMap;
use hand_indexer::hand_index::hand_indexer_t;

use rust_poker::card_range::CardRange;
use rayon::prelude::*;
use crate::state::BettingRound;

pub struct ISOMORPHIC {
    hand_indexer: hand_indexer_t
    // map of hand_index -> cluster index
    // (one for each player)
    // cluster_map: [HashMap<u64, usize>; 2],
    // number of differ clusters per player
    // pub count: [usize; 2],
}

pub struct EMD {

}

pub struct OCHS {

}

pub trait CardAbstraction {
    type AbsType;
    /**
     * round: the round we are using the abstraction for
     */
    fn init(round: BettingRound) -> Self::AbsType;
    fn get_cluster(&self, private_cards: (u8, u8), board_mask: u64) -> u64;
}

impl CardAbstraction for ISOMORPHIC {
    type AbsType = ISOMORPHIC;
    fn init(round: BettingRound) -> ISOMORPHIC {
        let hand_indexer = match round {
            BettingRound::Flop => hand_indexer_t::init(2, vec![2, 3]),
            BettingRound::Turn => hand_indexer_t::init(2, vec![2, 4]),
            BettingRound::River => hand_indexer_t::init(2, vec![2, 5])
        };
        ISOMORPHIC {
            hand_indexer: hand_indexer
        }
    }
    fn get_cluster(&self, private_cards: (u8, u8), mut board_mask: u64) -> u64 {
        let mut cards: Vec<u8> = Vec::new();
        cards.push(private_cards.0);
        cards.push(private_cards.1);
        while board_mask != 0 {
            let c: u8 = board_mask.trailing_zeros() as u8;
            cards.push(c);
            board_mask ^= 1u64 << c;
        }
        return self.hand_indexer.get_index(&cards);
    }
}

// impl CardAbstraction for ISOMORPHIC {
//     type AbsType = ISOMORPHIC;
// 
//     fn init(round: BettingRound, initial_board: u64, hand_ranges: &[CardRange]) -> ISOMORPHIC {
//         let (hand_indexer, n_board_cards, total_cards) = match round {
//             BettingRound::Flop => (hand_indexer_t::init(2, vec![2, 3]), initial_board.count_ones() as usize, 5),
//             BettingRound::Turn => (hand_indexer_t::init(2, vec![2, 4]), initial_board.count_ones() as usize, 6),
//             BettingRound::River => (hand_indexer_t::init(2, vec![2, 5]), initial_board.count_ones() as usize, 7)
//         };
//         let mut cluster_map: [HashMap<u64, usize>; 2] = [
//             HashMap::new(),
//             HashMap::new(),
//         ];
//         let mut count_arr = [0usize; 2];
// 
//         let mut cards = vec![0u8; total_cards];
//         // copy board cards to cards
//         let mut board = initial_board;
//         for i in 0..n_board_cards {
//             let c = board.trailing_zeros();
//             cards[i+2] = c as u8;
//             board ^= 1u64 << c;
//         }
//         crossbeam::scope(|scope| {
//             for (player, (map, count)) in cluster_map.chunks_mut(2).zip(count_arr.chunks_mut(2)).enumerate() {
//                 let (tx, rx) = crossbeam::channel::bounded::<u64>(100);
// 
//                 // consumer
//                 scope.spawn(move |_| {
//                     let mut c = 0;
//                     for index in rx {
//                         if !map[0].contains_key(&index) {
//                             map[0].insert(index, c);
//                             c += 1;
//                         }
//                     }
//                     count[0] = c;
//                 });
// 
//                 // spawn producers
//                 for slice in hand_ranges[player].hands.chunks(8) {
//                     let tx = tx.clone();
//                     let mut cards = cards.clone();
//                     scope.spawn(move |_| {
//                         for c in slice {
//                             let hand_mask = (1u64 << c.0) | (1u64 << c.1);
//                             if (hand_mask & initial_board) != 0 {
//                                 continue;
//                             }
//                             cards[0] = c.0;
//                             cards[1] = c.1;
//                             match total_cards - n_board_cards - 2 {
//                                 0 => {
//                                     let index = hand_indexer.get_index(&cards);
//                                     tx.send(index).unwrap();
//                                 },
//                                 1 => {
//                                     let used_card_mask = hand_mask | initial_board;
//                                     for i in 0u8..52 {
//                                         if ((1u64 << i) & used_card_mask) != 0 {
//                                             continue;
//                                         }
//                                         cards[5] = i;
//                                         let index = hand_indexer.get_index(&cards);
//                                         tx.send(index).unwrap();
//                                     }
// 
//                                 },
//                                 2 => {
//                                     let used_card_mask = hand_mask | initial_board;
//                                     for i in 0u8..52 {
//                                         if ((1u64 << i) & used_card_mask) != 0 {
//                                             continue;
//                                         }
//                                         cards[5] = i;
//                                         let used_card_mask = (1u64 << i) | used_card_mask;
//                                         for j in 0..i {
//                                             if ((1u64 << j) & used_card_mask) != 0 {
//                                                 continue;
//                                             }
//                                             cards[6] = j;
//                                             let index = hand_indexer.get_index(&cards);
//                                             tx.send(index).unwrap();
//                                         }
//                                      }
//                                  },
//                                 _ => { panic!("invalid card count"); }
//                             }
// 
//                         }
//                     });
//                 }
//             }
//         }).unwrap();
// 
//         ISOMORPHIC {
//             hand_indexer: hand_indexer,
//             count: count_arr,
//             cluster_map,
//             round: round,
//         }
//     }
//     fn get_index(&self, hand: &u64) -> usize {
//         return 0;
//     }
// }

// impl CardAbstraction for EMD {
//     fn init() {}
// }
// 
// impl CardAbstraction for OCHS {
//     fn init() {}
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use test::Bencher;
//     use crate::options;
//     use std::time::{Duration, Instant};
// 
//     #[bench]
//     fn bench_turn(b: &mut Bencher) {
//         // best score 9,125,032 ns/iter
//         // best with 1000 buffer channel and 8 producers
//         let round = BettingRound::Turn;
//         let flop_mask: u64 = 0b111;
//         let hand_ranges = CardRange::from_str_arr(["random", "random"].to_vec());
//         b.iter(|| {
//             let card_abs = ISOMORPHIC::init(round, flop_mask, &hand_ranges);
//             assert_eq!(card_abs.count[0], 12888);
//         });
//     }
// 
//     #[test]
//     fn time_river() {
//         // best score 104853038 ns
//         let round = BettingRound::River;
//         let flop_mask: u64 = 0b111;
//         let hand_ranges = CardRange::from_str_arr(["random", "random"].to_vec());
//         let start = Instant::now();
//         let card_abs = ISOMORPHIC::init(round, flop_mask, &hand_ranges);
//         let duration = start.elapsed().subsec_nanos();
//         println!("{}", duration);
//         assert!(duration < 104853038);
//     }
// }
