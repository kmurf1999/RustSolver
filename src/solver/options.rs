/**
 * Post Flop Solver options
 */

use rust_poker::card_range::CardRange;
use crate::state::{BettingRound, GameState, PlayerState};
use crate::action_abstraction::{ActionAbstraction};

#[derive(Debug)]
pub struct Options {
    pub n_players: usize,
    // hand range for each player
    pub hand_ranges: Vec<CardRange>,
    // starting stack for each player
    pub stack_sizes: Vec<u32>,
    // public card board mask
    pub board_mask: u64,
    // initial size of pot
    pub starting_pot: u32,
    // if bet or raise is > than this threshold
    // relative to stack size, go all in
    pub all_in_threshold: f32,
    // max length of each should be 3
    pub action_abstraction: ActionAbstraction,
    // 2 -> max 3-bet
    // 3 -> max 4-bet
    pub max_raises: u8
}

impl Options {
    // TODO implement to from trait
    pub fn to_state(&self) -> GameState {
        GameState {
            players: [
                PlayerState::init(self.stack_sizes[0]),
                PlayerState::init(self.stack_sizes[1])
            ],
            round: match self.board_mask.count_ones() {
                3 => BettingRound::Flop,
                4 => BettingRound::Turn,
                5 => BettingRound::River,
                _ => { panic!("invalid board mask"); }
            },
            current: 0,
            bets_settled: false,
            pot: self.starting_pot,
            raise_count: 0
        }
    }
}

pub fn default_river() -> Options {
    Options {

        n_players: 2,
        stack_sizes: vec![500, 500],
        board_mask: 0b11111,
        starting_pot: 35,
        all_in_threshold: 0.67,
        max_raises: 2,

        hand_ranges: vec![
            CardRange::from_str("random"),
            CardRange::from_str("random")
        ],

        action_abstraction: ActionAbstraction {
            bet_sizes: vec![
                vec![0.5, 1.0, 2.0]
            ],
            raise_sizes: vec![
                vec![3.0],
            ]
        },

    }
}
