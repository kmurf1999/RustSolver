use std::collections::HashMap;
use hand_indexer::hand_index::hand_indexer_t;

use rust_poker::card_range::CardRange;

use crate::state::BettingRound;


pub struct ISOMORPHIC {
    round: BettingRound,
    hand_indexer: hand_indexer_t,
    // map of hand_index -> cluster index
    // (one for each player)
    cluster_map: [HashMap<u64, usize>; 2],
    // number of differ clusters per player
    pub count: [usize; 2],
}

pub struct EMD {

}

pub struct OCHS {

}

pub trait CardAbstraction {
    type AbsType;
    /**
     * round: the round we are using the abstraction for
     * initial_board: the initial board of the simulation
     * ranges: a card range for each player
     */
    fn init(round: BettingRound, initial_board: u64, ranges: &[CardRange]) -> Self::AbsType;
    fn get_index(&self, hand: &u64) -> usize;
}

impl CardAbstraction for ISOMORPHIC {
    type AbsType = ISOMORPHIC;

    fn init(round: BettingRound, initial_board: u64, hand_ranges: &[CardRange]) -> ISOMORPHIC {
        let (hand_indexer, n_board_cards, total_cards) = match round {
            BettingRound::Flop => (hand_indexer_t::init(2, vec![2, 3]), initial_board.count_ones() as usize, 5),
            BettingRound::Turn => (hand_indexer_t::init(2, vec![2, 4]), initial_board.count_ones() as usize, 6),
            BettingRound::River => (hand_indexer_t::init(2, vec![2, 5]), initial_board.count_ones() as usize, 7)
        };
        let mut count = [0; 2];
        let mut cluster_map: [HashMap<u64, usize>; 2] = [
            HashMap::new(), HashMap::new()
        ];
        for player in &[0usize, 1] {
            // for getting indexes
            let mut cards = vec![0u8; total_cards];
            // copy board cards to cards
            let mut board = initial_board;
            for i in 0..n_board_cards {
                let c = board.trailing_zeros();
                cards[i+2] = c as u8;
                board ^= 1u64 << c;
            }
            for c in &hand_ranges[*player].hands {
                let hand_mask = (1u64 << c.0) | (1u64 << c.1);
                // conflics with initial board
                if (hand_mask & initial_board) != 0 {
                    continue;
                }
                cards[0] = c.0;
                cards[1] = c.1;
                match (total_cards - n_board_cards - 2) {
                    0 => {
                        let index = hand_indexer.get_index(&cards);
                        if !cluster_map[*player].contains_key(&index) {
                            cluster_map[*player].insert(index, count[*player]);
                            count[*player] += 1;
                        }
                    },
                    1 => {
                        // iterate over all valid next cards
                        let used_card_mask = hand_mask | initial_board;
                        for i in 0u8..52 {
                            if ((1u64 << i) & used_card_mask) != 0 {
                                continue;
                            }
                            cards[5] = i;
                            let index = hand_indexer.get_index(&cards);
                            if !cluster_map[*player].contains_key(&index) {
                                cluster_map[*player].insert(index, count[*player]);
                                count[*player] += 1;
                            }
                        }

                    },
                    2 => {
                        // iterate over all valid next cards
                        let used_card_mask = hand_mask | initial_board;
                        for i in 0u8..52 {
                            if ((1u64 << i) & used_card_mask) != 0 {
                                continue;
                            }
                            cards[5] = i;
                            let used_card_mask = used_card_mask | (1u64 << i);
                            for j in 0..i {
                                if ((1u64 << j) & used_card_mask) != 0 {
                                    continue;
                                }
                                cards[6] = j;
                                let index = hand_indexer.get_index(&cards);
                                if !cluster_map[*player].contains_key(&index) {
                                    cluster_map[*player].insert(index, count[*player]);
                                    count[*player] += 1;
                                }
                            }
                        }
                    }
                    _ => {
                        panic!("Invalid number of board cards");
                    }
                }
            }
        }

        ISOMORPHIC {
            hand_indexer: hand_indexer,
            count: count,
            cluster_map,
            round: round,
        }
    }
    fn get_index(&self, hand: &u64) -> usize {
        return 0;
    }
}

// impl CardAbstraction for EMD {
//     fn init() {}
// }
// 
// impl CardAbstraction for OCHS {
//     fn init() {}
// }
