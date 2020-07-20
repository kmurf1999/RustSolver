use crate::action_abstraction::Action;
use crate::state::BettingRound;

pub struct ActionNode {
    pub actions: Vec<Action>,
    pub index: usize,
    pub player: u8,
    pub round_idx: u8
}

/**
 * to be evaluated by cfr function
 * type: ALLIN, UNCONTESTED, SHOWDOWN
 */
pub enum TerminalType {
    ALLIN,
    SHOWDOWN,
    UNCONTESTED
}
impl TerminalType {
    pub fn to_string(&self) -> String {
        return match self {
            TerminalType::ALLIN => String::from("ALLIN"),
            TerminalType::SHOWDOWN => String::from("SHOWDOWN"),
            TerminalType::UNCONTESTED => String::from("UNCONTESTED"),
        }
    }
}

pub struct TerminalNode {
    pub value: u32, // size of pot
    pub ttype: TerminalType,
    pub last_to_act: u8, // 0 or 1
    pub round: BettingRound
}

pub enum GameTreeNode {
    Action(ActionNode),
    Terminal(TerminalNode),
    PrivateChance,
    PublicChance,
}
