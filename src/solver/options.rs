/**
 * Post Flop Solver options
 */

use rust_poker::card_range::CardRange;
use crate::state::{BettingRound, GameState, PlayerState};

#[derive(Debug)]
pub struct Options {
    // hand range for each player
    pub hand_ranges: [CardRange; 2],
    // starting stack for each player
    pub stack_sizes: [u32; 2],
    // public card board mask
    pub board_mask: u64,
    pub round: BettingRound,
    // initial size of pot
    pub starting_pot: u32,
    // if bet or raise is > than this threshold
    // relative to stack size, go all in
    pub all_in_threshold: f32,
    // max length of each should be 3
    pub flop_bet_sizes: Vec<f32>,
    pub turn_bet_sizes: Vec<f32>,
    pub river_bet_sizes: Vec<f32>,
    // max length of each should be two
    pub flop_raise_sizes: Vec<f32>,
    pub turn_raise_sizes: Vec<f32>,
    pub river_raise_sizes: Vec<f32>,
    // maximum number of raises per street
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
            current: 0,
            bets_settled: false,
            pot: self.starting_pot,
            round: self.round,
            board_mask: self.board_mask,
            raise_count: 0
        }
    }
}

pub fn default_options() -> Options {
    Options {

        hand_ranges: [
            CardRange::from_str("random"),
            CardRange::from_str("random")
        ],

        stack_sizes: [500, 500],
        board_mask: 0b11111,
        round: BettingRound::River,
        starting_pot: 45,
        all_in_threshold: 0.67,

        flop_bet_sizes: vec![0.5, 1.0],
        turn_bet_sizes: vec![0.5, 1.0],
        river_bet_sizes: vec![0.5, 1.0, 2.0],

        flop_raise_sizes: vec![3.0],
        turn_raise_sizes: vec![3.0],
        river_raise_sizes: vec![3.0],

        max_raises: 2
    }
}
