use crate::action_abstraction::{ActionAbstraction, Action};
use crate::constants::*;
use crate::options::Options;


#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum BettingRound {
    // Preflop,
    Flop,
    Turn,
    River
}

impl BettingRound {
    pub fn to_usize(&self) -> usize {
        return match self {
            BettingRound::Flop => 0,
            BettingRound::Turn => 1,
            BettingRound::River => 2
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PlayerState {
    stack : u32,
    wager: u32,
    has_folded: bool
}

impl PlayerState {
    pub fn init(stack: u32) -> PlayerState {
        PlayerState {
            stack: stack,
            wager: 0,
            has_folded: false
        }

    }
}

#[derive(Debug, Copy, Clone)]
pub struct GameState {
    pub players: [PlayerState; MAX_PLAYERS],
    pub pot: u32,
    pub raise_count: u8,
    pub current: u8,
    pub round: BettingRound,
    pub bets_settled: bool,
}

impl From<&Options> for GameState {
    fn from(options: &Options) -> Self {
        GameState {
            players: [
                PlayerState::init(options.stack_sizes[0]),
                PlayerState::init(options.stack_sizes[1]),
            ],
            round: match options.board_mask.count_ones() {
                3 => BettingRound::Flop,
                4 => BettingRound::Turn,
                5 => BettingRound::River,
                _ => panic!("invalid board mask")
            },
            current: 0,
            bets_settled: false,
            pot: options.starting_pot,
            raise_count: 0
        }
    }
}

impl GameState {
    fn current_player(&self) -> &PlayerState {
        return &self.players[usize::from(self.current)];
    }
    fn current_player_mut(&mut self) -> &mut PlayerState {
        return &mut self.players[usize::from(self.current)];
    }
    fn other_player(&self) -> &PlayerState {
        return &self.players[1 - usize::from(self.current)];
    }
    // fn other_player_mut(&mut self) -> &mut PlayerState {
    //     return &mut self.players[1 - usize::from(self.current)];
    // }
    pub fn is_uncontested(&self) -> bool {
        for p in &self.players {
            if p.has_folded {
                return true;
            }
        }
        return false;
    }
    pub fn is_terminal(&self) -> bool {
        return self.round == BettingRound::River
            || self.is_allin()
            || self.is_uncontested();
    }
    pub fn is_allin(&self) -> bool {
        for p in &self.players {
            if p.stack == 0 {
                return true;
            }
        }
        return false;
    }
    pub fn to_next_street(&self) -> GameState {
        let mut new_state = self.clone();
        new_state.bets_settled = false;
        new_state.current = 0;
        // reset wagers
        new_state.players.iter_mut().for_each(|p| p.wager = 0);
        match self.round {
            BettingRound::Flop => {
                new_state.round = BettingRound::Turn;
            },
            BettingRound::Turn => {
                new_state.round = BettingRound::River;
            },
            _ => panic!("Should not get here")
        }
        return new_state;
    }
    pub fn valid_actions(&self, action_abs: &ActionAbstraction) -> Vec<Action> {
        let mut actions: Vec<Action> = Vec::new();

        // TODO 
        let round = match action_abs.bet_sizes.len() {
            1 => self.round.to_usize() - 2,
            2 => self.round.to_usize() - 1,
            3 => self.round.to_usize(),
            _ => panic!("")
        };

        if self.other_player().wager == 0 {
            actions.push(Action::Check);
        }
        if self.other_player().wager > self.current_player().wager {
            actions.push(Action::Call);
        }
        if self.other_player().wager > self.current_player().wager {
            actions.push(Action::Fold);
        }
        if self.other_player().wager == 0 {
            for bet_size in &action_abs.bet_sizes[round] {
                let chips = bet_size * self.pot as f64;
                actions.push(Action::Bet(*bet_size));
                if chips > (ALLIN_THRESHOLD * self.current_player().stack as f64) {
                    break;
                }
            }
        }
        if self.raise_count < MAX_RAISES && !self.is_allin() && self.other_player().wager > self.current_player().wager {
            for raise_size in &action_abs.raise_sizes[round] {
                let chips = raise_size * self.other_player().wager as f64;
                actions.push(Action::Raise(*raise_size));
                if chips > (ALLIN_THRESHOLD * self.current_player().stack as f64) {
                    break;
                }
            }
        }

        return actions;
    }
    pub fn apply_action(&self, action: &Action) -> GameState {
        let mut new_state = self.clone();
        match action {
            Action::Bet(amt) => {
                let mut chips = (new_state.pot as f64 * amt) as u32;
                if chips > (new_state.current_player().stack as f64 * ALLIN_THRESHOLD) as u32 {
                    chips = new_state.current_player().stack;
                }
                new_state.current_player_mut().stack -= chips;
                new_state.current_player_mut().wager = chips;
                new_state.pot += chips;
                new_state.current = 1 - new_state.current;
            },
            Action::Raise(amt) => {
                let mut chips = (new_state.other_player().wager as f64 * amt) as u32;
                if chips > (new_state.current_player().stack as f64 * ALLIN_THRESHOLD) as u32 {
                    chips = new_state.current_player().stack;
                }
                new_state.current_player_mut().stack -= chips;
                new_state.current_player_mut().wager += chips;
                new_state.raise_count += 1;
                new_state.pot += chips;
                new_state.current = 1 - new_state.current;
            },
            Action::Call => {
                // TODO if more than two players
                // return chips back to player
                let wager_diff = self.other_player().wager
                    - new_state.current_player().wager;
                if new_state.current_player().stack >= wager_diff {
                    new_state.pot += wager_diff;
                    new_state.current_player_mut().stack -= wager_diff;
                } else {
                    new_state.pot += new_state.current_player().stack;
                    new_state.current_player_mut().stack = 0;
                }
                new_state.bets_settled = true;
            },
            Action::Check => {
                new_state.current = 1 - new_state.current;
                if usize::from(new_state.current) == MAX_PLAYERS - 1 {
                    new_state.bets_settled = true;
                }
            },
            Action::Fold => {
                new_state.current_player_mut().has_folded = true;
                // subtract other player bet from pot
                // TODO won't work if more than two players
                let wager_diff = self.other_player().wager
                    - new_state.current_player().wager;
                new_state.pot -= wager_diff;
                new_state.bets_settled = true;
            }
        }
        return new_state;
    }
}
