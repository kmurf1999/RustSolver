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
use rust_poker::constants::{CARD_COUNT, RANK_TO_CHAR, SUIT_TO_CHAR};
use rust_poker::hand_evaluator::{Hand, CARDS, evaluate};

use rayon::prelude::*;

use crate::tree::{Tree, NodeId};
use crate::nodes::{TerminalType};
use crate::tree_builder::build_game_tree;
use crate::infoset::{Infoset, InfosetTable, create_infosets};
use crate::nodes::GameTreeNode;
use crate::options::Options;
use crate::card_abstraction::{CardAbstraction, ISOMORPHIC, EMD, ICardAbstraction};
use crate::state::BettingRound;

#[derive(Debug, Copy, Clone)]
struct TrainHand {
    pub hands: [HoleCards; 2],
    pub board: [u8; 7]
}

impl TrainHand {
    /// Returns eval repr of board
    fn get_hand(&self, player: u8) -> Hand {
        let mut hand = Hand::empty();
        hand += CARDS[usize::from(self.hands[usize::from(player)].0)];
        hand += CARDS[usize::from(self.hands[usize::from(player)].1)];
        for i in 2..7 {
            hand += CARDS[usize::from(self.board[i])]
        }
        return hand;
    }
}

fn generate_possible_next_deals(round: BettingRound, hand: &TrainHand) -> Vec<u8> {
    let mut used_cards_mask = 0u64;
    // current number of board cards (before deal)
    let n_board_cards = match round {
        BettingRound::Flop => panic!("invalid number of board cards"),
        BettingRound::Turn => 3,
        BettingRound::River => 4
    };
    for i in 0..2 {
        used_cards_mask |= (1u64 << hand.hands[i].0) | (1u64 << hand.hands[i].1);
    }
    for i in 0..n_board_cards {
        used_cards_mask |= 1u64 << hand.board[i+2];
    }
    let mut possible_cards: Vec<u8> = Vec::new();
    for i in 0..CARD_COUNT {
        if (1u64 << i) & used_cards_mask == 0 {
            possible_cards.push(i);
        }
    }
    return possible_cards;
}

/// get all possible hole card combos
fn generate_all_hole_card_combos(mut board_mask: u64, hand_ranges: &[HandRange]) -> Vec<TrainHand> {

    // copy board
    let mut board = [0u8; 7];
    let mut i = 2;
    while board_mask.count_ones() > 0 {
        board[i] = board_mask.trailing_zeros() as u8;
        board_mask ^= 1u64 << board_mask.trailing_zeros();
        i += 1;
    }

    let mut combos: Vec<TrainHand> = Vec::new();
    for ci in &hand_ranges[0].hands {
        let ci_mask = (1u64 << ci.0) | (1u64 << ci.1);
        for cj in &hand_ranges[1].hands {
            let cj_mask = (1u64 << cj.0) | (1u64 << cj.1);
            if ci_mask & cj_mask == 0 {
                combos.push(TrainHand {
                    board: board.clone(),
                    hands: [*ci, *cj]
                });
            }
        }
    }
    return combos;
}

fn generate_hand<R: Rng>(rng: &mut R, mut board_mask: u64, hand_ranges: &[HandRange]) -> TrainHand {

    let mut used_cards_mask = board_mask;
    let mut board = [0u8; 7];
    let mut i = 2;
    let card_dist = Uniform::from(0..52);

    // copy board cards from mask
    while board_mask.count_ones() > 0 {
        board[i] = board_mask.trailing_zeros() as u8;
        board_mask ^= 1u64 << board_mask.trailing_zeros();
        i += 1;
    }

    // fill rest randomly
    while i < 7 {
        let c = card_dist.sample(rng);
        if ((1u64 << c) & used_cards_mask) == 0 {
            board[i] = c;
            used_cards_mask |= 1u64 << c;
            i += 1;
        }
    }

    let mut hands = [HoleCards(0, 0); 2];

    for i in 0..2 {
        loop {
            // get combo
            let c = hand_ranges[i].hands.choose(rng).unwrap();
            let combo_mask = (1u64 << c.0) | (1u64 << c.1);
            if (combo_mask & used_cards_mask) == 0 {
                used_cards_mask |= combo_mask;
                hands[i] = c.clone();
                break;
            }
        }
    }

    TrainHand {
        board,
        hands
    }
}


/**
 * A structure to implement monte carlo cfr
 */
#[derive(Debug)]
pub struct MCCFRTrainer {
    infosets: InfosetTable,
    game_tree: Tree<GameTreeNode>,
    card_abs: Vec<CardAbstraction>,
    hand_ranges: Vec<HandRange>,
    initial_board_mask: u64,
}

impl MCCFRTrainer {
    pub fn init(options: Options) -> Self {

        let mut hand_ranges = options.hand_ranges.to_owned();

        remove_invalid_combos(&mut hand_ranges, options.board_mask);

        let (n_actions, game_tree) = build_game_tree(&options);

        let card_abs = vec![
            // CardAbstraction::EMD(EMD::init(&hand_ranges, options.board_mask, BettingRound::Flop)),
            // CardAbstraction::ISOMORPHIC(ISOMORPHIC::init(&hand_ranges, options.board_mask, BettingRound::Flop)),
            // CardAbstraction::ISOMORPHIC(ISOMORPHIC::init(&hand_ranges, options.board_mask, BettingRound::Turn)),
            CardAbstraction::ISOMORPHIC(ISOMORPHIC::init(&hand_ranges, options.board_mask, BettingRound::River)),
        ];

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
        const PRUNE_THRESHOLD: usize = 10_000_000;
        /// number of iterations between discounts
        // const DISCOUNT_INTERVAL: usize = 1_000_000;
        const DISCOUNT_INTERVAL: usize = 100_000;
        const DISCOUNT_CAP: usize = 20_000_000;
        const N_THREADS: usize = 8;

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
                                a_self.hand_ranges.as_slice());

                        let q: f32 = rng.gen();

                        for player in 0..2 {
                            // a_self.cfr(0, player, hand, 1f32, true);
                            // println!("iteration {}", t.load());
                            if t.load() > PRUNE_THRESHOLD && q > 0.05 {
                                a_self.mccfr(&mut rng, 0, player, hand, 1f32, true);
                            } else {
                                a_self.mccfr(&mut rng, 0, player, hand, 1f32, false);
                            }
                        }

                        t.fetch_add(1);
                    }
                });
            }

            let a_self = a_self.clone();
            scope.spawn(move |_| {
                let mut threshold = DISCOUNT_INTERVAL;
                while t.load() < iterations {

                    let onems = time::Duration::from_millis(1);
                    thread::sleep(onems);

                    let tc = t.load();
                    if tc > DISCOUNT_CAP {
                        break;
                    }
                    if tc > threshold {
                        println!("calc br");
                        let br = a_self.calc_br();
                        println!("{} {}", br[0], br[1]);

                        let p = (tc / DISCOUNT_INTERVAL) as f32;
                        let d = p / (p + 1.0);
                        for i in 0..a_self.infosets.len() {
                            for j in 0..a_self.infosets[i].len() {
                                let infoset_mut = (&a_self.infosets[i][j] as *const Infoset) as *mut Infoset;
                                let n_actions = unsafe { (*infoset_mut).regrets.len() };
                                for k in 0..n_actions {
                                    unsafe {
                                        (*infoset_mut).regrets[k] = ((*infoset_mut).regrets[k] as f32 * d) as i32;
                                        (*infoset_mut).strategy_sum[k] = ((*infoset_mut).strategy_sum[k] as f32 * d) as i32;
                                    }
                                }
                            }
                        }
                        threshold = t.load() + DISCOUNT_INTERVAL;
                    }
                }
            });

        }).unwrap();

        // let mut rng = SmallRng::from_rng(thread_rng).unwrap();
        // let mut cards = generate_hand(
        //         &mut rng,
        //         a_self.initial_board_mask,
        //         a_self.hand_ranges.as_slice());

        // match &a_self.game_tree.get_node(3).data {
        //     GameTreeNode::Action(an) => {
        //         println!("an index {}", an.index);
        //         for combo in &a_self.hand_ranges[usize::from(an.player)].hands {
        //             cards.board[0] = combo.0;
        //             cards.board[1] = combo.1;
        //             let cluster_idx = match &a_self.card_abs[0] {
        //                 CardAbstraction::EMD(card_abs) => card_abs.get_cluster(&cards.board, an.player),
        //                 CardAbstraction::ISOMORPHIC(card_abs) => card_abs.get_cluster(&cards.board, an.player),
        //                 _ => panic!("HERE")
        //             };
        //             let s = a_self.infosets[an.index][cluster_idx].read().unwrap();
        //             print!("{} | ", combo.to_string());
        //             for (i, action) in an.actions.iter().enumerate() {
        //                 print!("{} {:.3}, ", action.to_string(), s.regrets[i]);
        //             }
        //             println!("");
        //         }
        //     },
        //     _ => {}
        // }

    }

    fn mccfr<R: Rng>(&self,
            rng: &mut R, node_id: NodeId,
            player: u8, mut hand: TrainHand,
            cfr_reach: f32, prune: bool) -> f32 {

        let node = self.game_tree.get_node(node_id);
        match &node.data {
            GameTreeNode::PublicChance(_) => {
                // progress to next node
                return self.mccfr(rng, node.children[0], player, hand, cfr_reach, prune);
            },
            GameTreeNode::PrivateChance => {
                // progress to next node
                return self.mccfr(rng, node.children[0], player, hand, cfr_reach, prune);
            },
            GameTreeNode::Terminal(tn) => {
                match tn.ttype {
                    TerminalType::UNCONTESTED => {
                        if player == tn.last_to_act {
                            return -1.0 * (tn.value as f32);
                        } else {
                            return 1.0 * (tn.value as f32);
                        }
                    },
                    TerminalType::SHOWDOWN => {
                        let hands = [hand.get_hand(0), hand.get_hand(1)];
                        let scores = [evaluate(&hands[0]), evaluate(&hands[1])];
                        if scores[0] == scores[1] {
                            return 0.0;
                        }
                        if scores[usize::from(player)] > scores[usize::from(1-player)] {
                            return 1.0 * (tn.value as f32);
                        } else {
                            return -1.0 * (tn.value as f32);
                        }
                    },
                    TerminalType::ALLIN => {
                        // evaluate as showdown
                        let hands = [hand.get_hand(0), hand.get_hand(1)];
                        let scores = [evaluate(&hands[0]), evaluate(&hands[1])];
                        if scores[0] == scores[1] {
                            return 0.0;
                        }
                        if scores[usize::from(player)] > scores[usize::from(1-player)] {
                            return 1.0 * (tn.value as f32);
                        } else {
                            return -1.0 * (tn.value as f32);
                        }
                    }
                }
            },
            GameTreeNode::Action(an) => {

                const PRUNE_THRESHOLD: i32 = -10000000;

                // get number of actions
                let n_actions = an.actions.len();

                // copy hole cards to board
                hand.board[0] = hand.hands[usize::from(an.player)].0;
                hand.board[1] = hand.hands[usize::from(an.player)].1;

                let cluster_idx = match &self.card_abs[usize::from(an.round_idx)] {
                    CardAbstraction::EMD(card_abs) => card_abs.get_cluster(&hand.board, an.player),
                    CardAbstraction::ISOMORPHIC(card_abs) => card_abs.get_cluster(&hand.board, an.player),
                    CardAbstraction::OCHS(card_abs) => card_abs.get_cluster(&hand.board, an.player)
                };

                {

                }
                if an.player == player {
                    let mut util = 0f32;
                    let mut utils = vec![0f32; n_actions];
                    let mut explored = vec![false; n_actions];

                    let infoset = &self.infosets[an.index][cluster_idx];
                    let strategy = infoset.get_strategy();

                    for i in 0..n_actions {
                        if prune {
                            if infoset.regrets[i] > PRUNE_THRESHOLD {
                            utils[i] = self.mccfr(
                                rng, node.children[i],
                                player, hand, cfr_reach, prune);
                            util += utils[i] * strategy[i];
                            explored[i] = true;
                            }
                        } else {
                        utils[i] = self.mccfr(
                            rng, node.children[i],
                            player, hand, cfr_reach, prune);
                        util += utils[i] * strategy[i];
                        }
                    }

                    // let cards = [
                    //     4u8 * 12 + 0,
                    //     4u8 * 0 + 0,
                    // ];

                    // if an.index == 0 && (hand.hands[usize::from(an.player)].0 == cards[0]) && (hand.hands[usize::from(an.player)].1 == cards[1]) {
                    //     for action in &an.actions {
                    //         print!("{} ", action.to_string());
                    //     }
                    //     println!("");
                    //     for i in 0..n_actions {
                    //         print!("{} ", infoset.regrets[i]);
                    //     }
                    //     println!("");
                    // }

                    

                    // update regrets
                    let infoset_mut = (infoset as *const Infoset) as *mut Infoset;
                    // let mut infoset_wlock = self.infosets[an.index][cluster_idx].write().unwrap();
                    // let strategy = infoset_wlock.get_strategy();

                    for i in 0..n_actions {
                        if prune {
                            if explored[i] {

                                // cap regrets
                                let mut new_regret = i64::from(infoset.regrets[i]) + 
                                    (100.0 * cfr_reach * (utils[i] - util)) as i64;
                                if new_regret > i32::MAX.into() {
                                    new_regret = i32::MAX.into();
                                } else if new_regret < i32::MIN.into() {
                                    new_regret = i32::MIN.into();
                                }
                                unsafe { (*infoset_mut).regrets[i] = new_regret as i32 };

                                let mut new_ssum = i64::from(infoset.strategy_sum[i]) +
                                    (100.0 * cfr_reach * strategy[i]) as i64;
                                if new_ssum > i32::MAX.into() {
                                    new_ssum = i32::MAX.into();
                                } else if new_ssum < i32::MIN.into() {
                                    new_ssum = i32::MIN.into();
                                }
                                unsafe { (*infoset_mut).strategy_sum[i] = new_ssum as i32 };
                            
                            }
                        } else {

                            // cap regrets
                            let mut new_regret = i64::from(infoset.regrets[i]) + 
                                (100.0 * cfr_reach * (utils[i] - util)) as i64;
                            if new_regret > i32::MAX.into() {
                                new_regret = i32::MAX.into();
                            } else if new_regret < i32::MIN.into() {
                                new_regret = i32::MIN.into();
                            }
                            unsafe { (*infoset_mut).regrets[i] = new_regret as i32 };

                            let mut new_ssum = i64::from(infoset.strategy_sum[i]) +
                                (100.0 * cfr_reach * strategy[i]) as i64;
                            if new_ssum > i32::MAX.into() {
                                new_ssum = i32::MAX.into();
                            } else if new_ssum < i32::MIN.into() {
                                new_ssum = i32::MIN.into();
                            }
                            unsafe { (*infoset_mut).strategy_sum[i] = new_ssum as i32 };

                        }
                    }

                    return util;
                } else {
                    // sample one action based on distribution
                    let infoset = &self.infosets[an.index][cluster_idx];
                    let strategy = infoset.get_strategy();
                    let dist = WeightedIndex::new(&strategy).unwrap();
                    let a_idx = dist.sample(rng);
                    return self.mccfr(
                        rng, node.children[a_idx],
                        player, hand, cfr_reach * strategy[a_idx], prune);
                }
            }
        }
    }

    fn cfr(&self, node_id: NodeId,
        player: u8, mut hand: TrainHand,
        cfr_reach: f32, prune: bool) -> f32 {
            let node = self.game_tree.get_node(node_id);
            match &node.data {
                GameTreeNode::PrivateChance => {
                    let hole_combos = generate_all_hole_card_combos(
                        self.initial_board_mask,
                        self.hand_ranges.as_slice()
                    );
                    let child_cfr_reach = cfr_reach * (1.0 / hole_combos.len() as f32);
                    let util = AtomicCell::new(0f32);
                    hole_combos.par_iter().for_each(|combo| {
                        let u = self.cfr(
                            node.children[0], player,
                            combo.clone(), child_cfr_reach, prune
                        );
                        util.store(util.load() + u);
                    });
                    return util.load();
                },
                GameTreeNode::PublicChance(pc) => {
                    let possible_deals =
                        generate_possible_next_deals(pc.round, &hand);
                    let next_index = match pc.round {
                        BettingRound::Flop => panic!("should not get here"),
                        BettingRound::Turn => 5,
                        BettingRound::River => 6
                    };
                    let child_cfr_reach = cfr_reach * (1.0 / possible_deals.len() as f32);
                    let util = AtomicCell::new(0f32);
                    possible_deals.par_iter().for_each(|card| {
                        let mut next_hand = hand.clone();
                        next_hand.board[next_index] = *card;
                        let u = self.cfr(
                            node.children[0], player,
                            next_hand, child_cfr_reach, prune
                        );
                        util.store(util.load() + u);
                    });
                    return util.load();
                },
                GameTreeNode::Terminal(tn) => {
                    match tn.ttype {
                        TerminalType::UNCONTESTED => {
                            if player == tn.last_to_act {
                                return -1.0 * (tn.value as f32);
                            } else {
                                return 1.0 * (tn.value as f32);
                            }
                        },
                        TerminalType::SHOWDOWN => {
                            let hands = [hand.get_hand(0), hand.get_hand(1)];
                            let scores = [evaluate(&hands[0]), evaluate(&hands[1])];
                            if scores[0] == scores[1] {
                                return 0.0;
                            }
                            if scores[usize::from(player)] > scores[usize::from(1-player)] {
                                return 1.0 * (tn.value as f32);
                            } else {
                                return -1.0 * (tn.value as f32);
                            }
                        },
                        TerminalType::ALLIN => {
                            // evaluate as showdown
                            let hands = [hand.get_hand(0), hand.get_hand(1)];
                            let scores = [evaluate(&hands[0]), evaluate(&hands[1])];
                            if scores[0] == scores[1] {
                                return 0.0;
                            }
                            if scores[usize::from(player)] > scores[usize::from(1-player)] {
                                return 1.0 * (tn.value as f32);
                            } else {
                                return -1.0 * (tn.value as f32);
                            }
                        }
                    }
                },
                GameTreeNode::Action(an) => {
                    let n_actions = an.actions.len();
                    // copy hole cards to board
                    hand.board[0] = hand.hands[usize::from(an.player)].0;
                    hand.board[1] = hand.hands[usize::from(an.player)].1;
                    let cluster_idx = match &self.card_abs[usize::from(an.round_idx)] {
                        CardAbstraction::EMD(card_abs) => card_abs.get_cluster(&hand.board, an.player),
                        CardAbstraction::ISOMORPHIC(card_abs) => card_abs.get_cluster(&hand.board, an.player),
                        CardAbstraction::OCHS(card_abs) => card_abs.get_cluster(&hand.board, an.player)
                    };


                    let mut util = 0f32;
                    let mut utils = vec![0f32; n_actions];
                    let infoset = &self.infosets[an.index][cluster_idx];
                    let strategy = infoset.get_strategy();

                    for i in 0..n_actions {
                        if an.player == player {
                            utils[i] = self.cfr(
                                node.children[i], player,
                                hand, cfr_reach, prune
                            );
                        } else {
                            utils[i] = self.cfr(
                                node.children[i], player,
                                hand, strategy[i] * cfr_reach, prune
                            );
                        }
                        util += utils[i] * strategy[i];
                    }

                    let cards = [
                        4u8 * 12 + 0,
                        4u8 * 0 + 0
                    ];
                    if an.index == 0 && (hand.hands[usize::from(an.player)].0 == cards[0]) && (hand.hands[usize::from(an.player)].1 == cards[1]) {
                        for action in &an.actions {
                            print!("{} ", action.to_string());
                        }
                        println!("");
                        for i in 0..n_actions {
                            print!("{} ", infoset.regrets[i]);
                        }
                        println!("");
                    }
                    


                    if an.player != player {
                        return util;
                    }

                    let infoset_mut = (infoset as *const Infoset) as *mut Infoset;
                    let strategy = infoset.get_strategy();
                    for i in 0..n_actions {
                        unsafe {
                            (*infoset_mut).regrets[i] +=
                                (10000.0 * cfr_reach * (utils[i] - util)) as i32;
                            (*infoset_mut).strategy_sum[i] +=
                                (10000.0 * cfr_reach * strategy[i]) as i32;
                        }
                    }

                    return util;

                }
            }
    }

    fn calc_br(&self) -> Vec<f32> {
        // 2: num player,
        let mut op = vec![vec![1.0; 1]; 2];
        let res = self.abstract_br(0, op);
        let mut out = vec![0f32; res.len()];
        for i in 0..res.len() {
            out[i] = res[i][0];
        }
        return out;
    }

    fn abstract_br(&self, curr_node: NodeId, op: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let node = self.game_tree.get_node(curr_node);
        match &node.data {
            GameTreeNode::Terminal(_) => {
                return self.abstract_br_terminal(curr_node, op);
            },
            GameTreeNode::PublicChance(_) => {
                return self.abstract_br(node.children[0], op);
            },
            GameTreeNode::PrivateChance => {
                return self.abstract_br(node.children[0], op);
            },
            _ => {
                return self.abstract_br_infoset(curr_node, op);
            }
        }
    }

    fn abstract_br_infoset(&self, curr_node: NodeId, op: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let node = self.game_tree.get_node(curr_node);
        match &node.data {
            GameTreeNode::Action(an) => {
                let info_idx = an.index;
                let n_buckets = match &self.card_abs[usize::from(an.round_idx)] {
                    CardAbstraction::ISOMORPHIC(card_abs) => card_abs.get_size(an.player),
                    CardAbstraction::EMD(card_abs) => card_abs.get_size(an.player),
                    CardAbstraction::OCHS(card_abs) => card_abs.get_size(an.player),
                };

                let mut probabilites: Vec<Vec<f32>> = Vec::new();
                for i in 0..n_buckets {
                    probabilites.push(self.infosets[info_idx][i].get_final_strategy());
                }

                let mut payoffs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(node.children.len());
                for a in 0..node.children.len() {
                    let mut newop: Vec<Vec<f32>> = op.clone();
                    for h in 0..newop[usize::from(an.player)].len() {
                        newop[usize::from(an.player)][h] *= probabilites[h][a];
                    }

                    payoffs.push(self.abstract_br(node.children[a], newop));
                }

                let opp = usize::from(1 - an.player);
                let mut max_val = payoffs[0][usize::from(an.player)][0];
                let mut max_index = 0usize;
                for a in 1..node.children.len() {
                    if max_val < payoffs[a][usize::from(an.player)][0] {
                        max_val = payoffs[a][usize::from(an.player)][0];
                        max_index = a;
                    }
                }

                let mut res: Vec<Vec<f32>> = vec![vec![0.0; 1]; 2];
                res[usize::from(an.player)][0] = max_val;
                res[opp][0] = payoffs[max_index][opp][0];
                return res;
            },
            _ => panic!("error")
        }
    }

    fn abstract_br_terminal(&self, curr_node: NodeId, op: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let node = self.game_tree.get_node(curr_node);
        match &node.data {
            GameTreeNode::Terminal(tn) => {
                let mut payoffs: Vec<Vec<f32>> = vec![vec![0.0; op[0].len()]; op.len()];
                let mut res: Vec<Vec<f32>> = vec![vec![0.0; 1]; op.len()];
                let money_f = tn.value as f32;

                match tn.ttype {
                    TerminalType::UNCONTESTED => {
                        let fold_player = tn.last_to_act;

                        let mut opp_ges_p = vec![0.0; 2];
                        for p in 0..op.len() {
                            let opp = 1 - p;
                            for g in 0..op[0].len() {
                                payoffs[p][g] = op[opp][g] * (if p == fold_player.into() { -1.0 } else { 1.0 }) * money_f;
                                res[p][0] += payoffs[p][g];
                                opp_ges_p[p] += op[opp][g];
                            }
                            res[p][0] *= 1.0 / opp_ges_p[p];
                        }
                        return res;
                    },
                    _ => {
                        let mut opp_ges_p = vec![0.0; 2];
                        for p in 0..op.len() {
                            let opp = 1 - p;
                            for g in 0..op[0].len() {
                                payoffs[p][g] = op[opp][g] * money_f;
                                res[p][0] += payoffs[p][g];
                                opp_ges_p[p] += op[opp][g];
                            }
                            res[p][0] *= 1.0 / opp_ges_p[p];
                        }
                        return res;
                    }
                }
            },
            _ => panic!("error")
        }
    }
}
