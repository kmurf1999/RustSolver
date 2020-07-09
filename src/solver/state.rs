use crate::actions::Action;

static ALLIN_THRESHOLD: f32 = 0.67;

// maximum raises per round
// 2 means 3-bet
// 3 means 4-bet
static MAX_RAISES: u8 = 2;

const MAX_PLAYERS: usize = 2;

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
    pub current: u8,
    pub bets_settled: bool,
    pub pot: u32,
    pub round: BettingRound,
    pub board_mask: u64,
    pub raise_count: u8
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
            _ => {
                // do nothing
                assert!(false);
            }
        }
        return new_state;
    }
    pub fn is_valid_action(&self, action: &Action) -> bool {
        match action {
            Action::Bet(_) => {
                // if other player hasn't bet
                // and we're not all in
                return self.other_player().wager == 0
                    && self.current_player().stack != 0;

            },
            Action::Raise(_) => {
                // other player has bigger wager than us
                return (self.current_player().wager
                    < self.other_player().wager)
                    && self.raise_count < MAX_RAISES
                    && self.current_player().stack != 0;
            },
            Action::Call => {
                // if other player has bet,
                // and current player isn't all in
                return (self.current_player().wager
                    < self.other_player().wager)
                    && self.current_player().stack != 0;
            },
            Action::Check => {
                // if other player hasn't bet
                return self.other_player().wager == 0;
            },
            Action::Fold => {
                // if other player has bet
                // and current player isn't all in
                return (self.other_player().wager
                    > self.current_player().wager)
                    && self.current_player().stack != 0;
            }
        }
    }
    pub fn apply_action(&self, action: &Action) -> GameState {
        let mut new_state = self.clone();
        match action {
            Action::Bet(amt) => {
                let mut chips = (new_state.pot as f32 * amt) as u32;
                if chips > (new_state.current_player().stack as f32 * ALLIN_THRESHOLD) as u32 {
                    chips = new_state.current_player().stack;
                }
                new_state.current_player_mut().stack -= chips;
                new_state.current_player_mut().wager = chips;
                new_state.pot += chips;
                new_state.current = 1 - new_state.current;
            },
            Action::Raise(amt) => {
                let mut chips = (new_state.other_player().wager as f32 * amt) as u32;
                if chips > (new_state.current_player().stack as f32 * ALLIN_THRESHOLD) as u32 {
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
                if usize::from(new_state.current) == MAX_PLAYERS - 1 {
                    new_state.bets_settled = true;
                }
                new_state.current = 1 - new_state.current;
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
