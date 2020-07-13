// use rust_poker::hand::Hand;
use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform};

use rust_poker::card_range::CardRange;
use rust_poker::combined_range::CombinedRange;
use rust_poker::equity_calc::{remove_invalid_combos};
use rust_poker::hand::{Hand, CARDS};
use rust_poker::evaluator::HAND_EVAL;

use crate::arena::{Arena, NodeId};
use crate::nodes::{TerminalType};
use crate::tree_builder::{TreeBuilder};
use crate::infoset::{Infoset, InfosetTable};
use crate::nodes::GameTreeNode;
use crate::options::Options;
use crate::card_abstraction::{CardAbstraction, ISOMORPHIC};
use crate::state::BettingRound;

struct TrainHand {
    board: [u8; 7], // keep first two 0, (used for faster indexing)
    hole_cards: Vec<(u8, u8)>
}
fn generate_hand<R: Rng>(rng: &mut R, mut board_mask: u64, hand_ranges: &Vec<CardRange>) -> TrainHand {
    let mut used_cards_mask = board_mask;
    let mut hole_cards = vec![(0u8, 0u8); hand_ranges.len()];
    for (i, range) in hand_ranges.iter().enumerate() {
        loop {
            // get combo
            let c = range.hands.choose(rng).unwrap();
            let combo_mask = (1u64 << c.0) | (1u64 << c.1);
            if (combo_mask & used_cards_mask) == 0 {
                used_cards_mask |= combo_mask;
                hole_cards[i] = (c.0, c.1);
                break;
            }
        }
    }
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
    TrainHand {
        board,
        hole_cards
    }
}


/**
 * A structure to implement monte carlo cfr
 */
pub struct MCCFRTrainer {
    infoset_table: InfosetTable,
    game_tree: Arena<GameTreeNode>,
    iteration: usize,
    // TODO maybe change to combined range
    board_mask: u64,
    hand_ranges: Vec<CardRange>,
    // TODO change to support all rounds and abs type
    card_abs: ISOMORPHIC
}

impl MCCFRTrainer {
    pub fn init(options: Options) -> Self {
        // build tree
        let mut tree_builder = TreeBuilder::init(&options);
        tree_builder.build();
        // create infoset table
        let infoset_table = InfosetTable::init(tree_builder.action_count());

        let mut hand_ranges = options.hand_ranges.to_owned();
        remove_invalid_combos(&mut hand_ranges, options.board_mask);

        MCCFRTrainer {
            infoset_table,
            game_tree: tree_builder.tree,
            hand_ranges,
            board_mask: options.board_mask,
            iteration: 0,
            card_abs: ISOMORPHIC::init(BettingRound::River)
        }
    }
    /**
     * iterations: number of iterations to train for
     */
    pub fn train<R: Rng>(&self, rng: &mut R, iterations: usize) {
        const PRUNE_THRESHOLD: usize = 5000;
        const DISCOUNT_INTERVAL: usize = 5000;
        for t in self.iteration..(self.iteration + iterations) {
            // generate hand
            let hand = generate_hand(rng, self.board_mask, &self.hand_ranges);
            for player in &[0u8, 1u8] {
                self.mccfr(rng, 0, *player, &hand, 1.0, false);
            }
            // run simulation for each player
            // perform discounting
            if t % DISCOUNT_INTERVAL == 0 {

            }
        }
    }
    fn mccfr<R: Rng>(&self,
            rng: &mut R, node_id: NodeId,
            player: u8, hand: &TrainHand,
            cfr_reach: f64, prune: bool) -> f64 {
        let node = self.game_tree.get_node(node_id);
        match &node.data {
            GameTreeNode::PublicChance => {
                return self.mccfr(rng, node.children[0], player, hand, cfr_reach, prune);
            },
            GameTreeNode::PrivateChance => {
                return self.mccfr(rng, node.children[0], player, hand, cfr_reach, prune);
            },
            GameTreeNode::Terminal(tn) => {
                match tn.ttype {
                    TerminalType::UNCONTESTED => {
                        if player == tn.last_to_act {
                            return -1.0 * tn.value as f64;
                        } else {
                            return tn.value as f64;
                        }
                    },
                    TerminalType::SHOWDOWN => {
                        let mut public_cards: Hand = Hand::empty();
                        for card in &hand.board {
                            public_cards += CARDS[*card as usize];
                        }
                        let p0_score = HAND_EVAL.evaluate(&(public_cards + Hand::from_hole_cards(hand.hole_cards[0].0, hand.hole_cards[0].1)));
                        let p1_score = HAND_EVAL.evaluate(&(public_cards + Hand::from_hole_cards(hand.hole_cards[0].0, hand.hole_cards[0].1)));
                        if p0_score == p1_score {
                            return 0.0;
                        }
                        if p0_score > p1_score {
                            return if player == 0 { 1.0 } else { -1.0 } * tn.value as f64;
                        } else {
                            return if player == 1 { 1.0 } else { -1.0 } * tn.value as f64;
                        }
                    },
                    TerminalType::ALLIN => {
                        return 0.0;
                        // return rough approximate equity
                    }
                }
                // evaluate terminal node
            },
            GameTreeNode::Action(an) => {

            }
        }
    }
}
