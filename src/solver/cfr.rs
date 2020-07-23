use std::iter::FromIterator;
use std::{thread, time};
use crossbeam::atomic::AtomicCell;
use std::sync::Arc;
use std::sync::Mutex;

use rand::{SeedableRng, thread_rng};
use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::distributions::WeightedIndex;
use rand::distributions::{Distribution, Uniform};

use rust_poker::hand_range::{HandRange, HoleCards};
use rust_poker::equity_calculator::{get_board_from_bit_mask, calc_equity, remove_invalid_combos};
use rust_poker::constants::{RANK_TO_CHAR, SUIT_TO_CHAR};
use rust_poker::hand_evaluator::{Hand, CARDS, evaluate};

use crate::tree::{Tree, NodeId};
use crate::nodes::{TerminalType};
use crate::tree_builder::build_game_tree;
use crate::infoset::{Infoset, InfosetTable, create_infosets};
use crate::nodes::GameTreeNode;
use crate::options::Options;
use crate::card_abstraction::{CardAbstraction, OCHS};
use crate::state::BettingRound;

#[derive(Debug, Copy, Clone)]
struct TrainHand {
    pub hands: [[u8; 7]; 2],
    // keep first two 0, (used for faster indexing
    // pub hands: [HoleCards; 2]
}

impl TrainHand {
    /// Returns eval repr of board
    fn get_hand(&self, player: u8) -> Hand {
        let mut hand = Hand::empty();
        for i in 0..7 {
            hand += CARDS[usize::from(self.hands[usize::from(player)][i])];
        }
        return hand;
    }
    // fn get_board_mask(&self, n_cards: usize) -> u64 {
    //     assert!(n_cards <= 5);
    //     let mut board_mask: u64 = 0;
    //     for i in 2..(n_cards+2) {
    //         board_mask |= 1u64 << self.board[i];
    //     }
    //     return board_mask;
    // }
}

fn generate_hand<R: Rng>(rng: &mut R, mut board_mask: u64, hand_ranges: &Vec<HandRange>) -> TrainHand {
    let mut used_cards_mask = board_mask;
    let mut board = [0u8; 7];
    let mut i = 2;
    let card_dist = Uniform::from(0..52);
    while board_mask.count_ones() > 0 {
        board[i] = board_mask.trailing_zeros() as u8;
        board_mask ^= 1u64 << board_mask.trailing_zeros();
        i += 1;
    }
    while i < 7 {
        let c = card_dist.sample(rng);
        if ((1u64 << c) & used_cards_mask) == 0 {
            board[i] = c;
            used_cards_mask |= 1u64 << c;
            i += 1;
        }
    }
    let mut hands = [[0u8; 7]; 2];
    for i in 0..2 {
        loop {
            // get combo
            let c = hand_ranges[i].hands.choose(rng).unwrap();
            let combo_mask = (1u64 << c.0) | (1u64 << c.1);
            if (combo_mask & used_cards_mask) == 0 {
                used_cards_mask |= combo_mask;
                hands[i] = board.clone();
                hands[i][0] = c.0;
                hands[i][1] = c.1;
                break;
            }
        }
    }

    TrainHand {
        hands
    }
}


/**
 * A structure to implement monte carlo cfr
 */
pub struct MCCFRTrainer {
    infosets: InfosetTable,
    game_tree: Tree<GameTreeNode>,
    card_abs: OCHS,
    hand_ranges: Vec<HandRange>,
    initial_board_mask: u64,
}

impl MCCFRTrainer {
    pub fn init(options: Options) -> Self {

        let mut hand_ranges = options.hand_ranges.to_owned();

        remove_invalid_combos(&mut hand_ranges, options.board_mask);

        let (n_actions, game_tree) = build_game_tree(&options);

        // River card abs
        let card_abs = OCHS::init(
            &options.hand_ranges,
            options.board_mask,
            BettingRound::River);

        // intialize infosets
        let infosets = create_infosets(n_actions, &game_tree, &card_abs);

        MCCFRTrainer {
            infosets,
            game_tree,
            hand_ranges,
            initial_board_mask: options.board_mask,
            card_abs
        }
    }
    /**
     * iterations: number of iterations to train for
     */
    pub fn train(&mut self, iterations: usize) {
        /// number of iterations before pruning
        const PRUNE_THRESHOLD: usize = 1_000_000;
        /// number of iterations between discounts
        const DISCOUNT_INTERVAL: usize = 10_000;
        const N_THREADS: usize = 6;

        let thread_rng = thread_rng();

        let t = Arc::new(AtomicCell::new(0));
        let a_self = Arc::new(self);
        crossbeam::scope(|scope| {
            for _ in 0..N_THREADS {
                let a_self = Arc::clone(&a_self);
                let mut rng = SmallRng::from_rng(thread_rng).unwrap();
                let t = t.clone();
                scope.spawn(move |_| {
                    while t.load() < iterations {

                        let hand = generate_hand(
                                &mut rng,
                                a_self.initial_board_mask,
                                &a_self.hand_ranges);

                        // println!("{} {} {} {} {} {} {}",
                        //          hand.hole_cards[1].0,
                        //          hand.hole_cards[1].1,
                        //          hand.board[2],
                        //          hand.board[3],
                        //          hand.board[4],
                        //          hand.board[5],
                        //          hand.board[6],
                        //          );


                        let q: f64 = rng.gen();
                        for player in &[0u8, 1u8] {
                            if t.load() > PRUNE_THRESHOLD && q > 0.05 {
                                a_self.mccfr(&mut rng, 0, *player, hand, 1f64, 1f64, true);
                            } else {
                                a_self.mccfr(&mut rng, 0, *player, hand, 1f64, 1f64, false);
                            }
                        }

                        t.fetch_add(1);
                    }
                });
            }

            // scope.spawn(move |_| {
            //     while t.load() < iterations {
            //         let onems = time::Duration::from_millis(1);
            //         thread::sleep(onems);
            //         if t.load() % DISCOUNT_INTERVAL == 0 {
            //             let p = (t.load() / DISCOUNT_INTERVAL) as f64;
            //             let d = p / (p + 1.0);
            //             for i in 0..a_self.infosets.len() {
            //                 for j in 0..a_self.infosets[i].len() {
            //                     let n_actions = a_self.infosets[i][j].regrets.len();
            //                     let infoset_mut = (&a_self.infosets[i][j] as *const Infoset) as *mut Infoset;
            //                     for k in 0..n_actions {
            //                         unsafe {
            //                             (*infoset_mut).regrets[k] *= d;
            //                             (*infoset_mut).strategy_sum[k] *= d;
            //                         }
            //                     }
            //                 }
            //             }

            //         }
            //     }
            // });
        }).unwrap();

        let mut rng = SmallRng::from_rng(thread_rng).unwrap();
        let mut cards = generate_hand(
                &mut rng,
                a_self.initial_board_mask,
                &a_self.hand_ranges).hands[0];

        match &a_self.game_tree.get_node(1).data {
            GameTreeNode::Action(an) => {
                for combo in &a_self.hand_ranges[usize::from(an.player)].hands {
                    cards[0] = combo.0;
                    cards[1] = combo.1;
                    let cluster_idx = a_self.card_abs.get_cluster(&cards, an.player);
                    let s = a_self.infosets[an.index][cluster_idx].get_final_strategy();
                    print!("{} | ", combo.to_string());
                    for (i, action) in an.actions.iter().enumerate() {
                        print!("{} {:.3}, ", action.to_string(), (s[i] * 1000.0).round() / 1000.0);
                    }
                    println!("");
                }
            },
            _ => {}
        }

    }
    fn mccfr<R: Rng>(&self,
            rng: &mut R, node_id: NodeId,
            player: u8, hand: TrainHand,
            p: f64, op: f64, prune: bool) -> f64 {

        let node = self.game_tree.get_node(node_id);
        match &node.data {
            GameTreeNode::PublicChance => {
                // progress to next node
                return self.mccfr(rng, node.children[0], player, hand, p, op, prune);
            },
            GameTreeNode::PrivateChance => {
                // progress to next node
                return self.mccfr(rng, node.children[0], player, hand, p, op, prune);
            },
            GameTreeNode::Terminal(tn) => {
                match tn.ttype {
                    TerminalType::UNCONTESTED => {
                        if player == tn.last_to_act {
                            return -1.0 * (tn.value as f64);
                        } else {
                            return 1.0 * (tn.value as f64);
                        }
                    },
                    TerminalType::SHOWDOWN => {
                        let hands = [hand.get_hand(0), hand.get_hand(1)];
                        let scores = [evaluate(&hands[0]), evaluate(&hands[1])];
                        if scores[0] == scores[1] {
                            return 0.0;
                        }
                        if scores[usize::from(player)] > scores[usize::from(1-player)] {
                            return 1.0 * (tn.value as f64);
                        } else {
                            return -1.0 * (tn.value as f64);
                        }
                    },
                    TerminalType::ALLIN => {
                        return 0.0;
                        //let n_public_cards = match tn.round {
                        //    BettingRound::Flop => 3,
                        //    BettingRound::Turn => 4,
                        //    BettingRound::River => panic!("Should not be All in node")
                        //};
                        //let board_mask = hand.get_board_mask(n_public_cards);
                        //let ranges: Vec<HandRange> = hand.hole_cards
                        //    .iter()
                        //    .map(|c| HandRange::from_string(c.to_string()))
                        //    .collect();
                        //// get approx equity
                        //let equities = calc_equity(&ranges, board_mask, 1, 1000);
                        //return equities[usize::from(player)];
                    }
                }
            },
            GameTreeNode::Action(an) => {

                const PRUNE_THRESHOLD: f64 = -5000.0;

                // get number of actions
                let n_actions = an.actions.len();

                let cluster_idx = self.card_abs.get_cluster(&hand.hands[usize::from(an.player)], an.player);

                let infoset = &self.infosets[an.index][cluster_idx];
                let strategy = infoset.get_strategy();

                if an.player == player {
                    let mut util = 0f64;
                    let mut utils = vec![0f64; n_actions];
                    let mut explored = vec![false; n_actions];

                    for i in 0..n_actions {
                        if prune {
                            if infoset.regrets[i] > PRUNE_THRESHOLD {

                               utils[i] = self.mccfr(
                                   rng, node.children[i],
                                   player, hand, p * strategy[i], op, prune);

                               util += utils[i] * strategy[i];
                               explored[i] = true;
                            }
                        } else {
                           utils[i] = self.mccfr(
                               rng, node.children[i],
                               player, hand, p * strategy[i], op, prune);

                           util += utils[i] * strategy[i];
                        }
                    }


                    // board_mask: get_card_mask("Kh5h7sJd3h"),
                    // let cards = [
                    //     4u8 * 9 + 0,
                    //     4u8 * 12 + 1,

                    //     4 * 11 + 1,
                    //     4 * 3 + 1,
                    //     4 * 5 + 0,
                    //     4 * 9 + 2,
                    //     4 * 1 + 1,
                    // ];

                    // let c = self.card_abs.get_cluster(&cards, an.player);
                    // if an.index == 0 && c == cluster_idx {
                    //     for action in &an.actions {
                    //         print!("{} ", action.to_string());
                    //     }
                    //     println!("");
                    //     for i in 0..n_actions {
                    //         print!("{} ", infoset.strategy_sum[i]);
                    //     }
                    //     // println!("");
                    //     // let s = infoset.get_final_strategy();
                    //     // for i in 0..n_actions {
                    //     //     print!("{} ", s[i]);
                    //     // }
                    //     println!("");
                    //     // println!("");
                    // }

                    // update regrets
                    let infoset_mut = (infoset as *const Infoset) as *mut Infoset;
                    for i in 0..n_actions {
                        if prune {
                            if explored[i] {
                                unsafe {
                                    (*infoset_mut).regrets[i] += utils[i] - util;
                                    (*infoset_mut).strategy_sum[i] += (1.0 / op) * p * strategy[i];
                                }
                            }
                        } else {
                            unsafe {
                                (*infoset_mut).regrets[i] += utils[i] - util;
                                (*infoset_mut).strategy_sum[i] += (1.0 / op) * p * strategy[i];
                            }
                        }
                    }

                    return util;
                } else {
                    // sample one action based on distribution
                    let dist = WeightedIndex::new(&strategy).unwrap();
                    let a_idx = dist.sample(rng);
                    // // let child_cfr_reach = strategy[a_idx] * cfr_reach;
                    return self.mccfr(
                        rng, node.children[a_idx],
                        player, hand, p, op * strategy[a_idx], prune);
                    // let mut utils = vec![0f64; n_actions];
                    // let mut util = 0f64;
                    // for i in 0..n_actions {
                    //     util += self.mccfr(
                    //         rng, node.children[i],
                    //         player, hand, p, op * strategy[i], prune);
                    // }
                    // return util;
                }
            }
        }
    }
}
