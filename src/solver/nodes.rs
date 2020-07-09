use rand::Rng;

use crate::actions::Action;

use rand::distributions::{Distribution, Uniform};


pub struct ActionNode {
    pub actions: Vec<Action>,
    pub index: usize,
    pub player: u8
}

/**
 * to be evaluated by cfr function
 * type: ALLIN, UNCONTESTED, SHOWDOWN
 */
pub enum TerminalNodeType {
    Allin,
    Uncontested,
    Showdown
}
pub struct TerminalNode {
    pub value: u32, // size of pot
    pub node_type: TerminalNodeType,
    pub last_to_act: u8, // 0 or 1
    pub board_mask: u64
}

pub struct PublicChanceNode {
    pub board_mask: u64
}

pub struct PrivateChanceNode {
    pub board_mask: u64
}

// public chance node
impl PublicChanceNode {
    // return one public chance outcome (new board mask)
    pub fn sample_one<R: Rng>(&self, rng: &mut R, used_cards_mask: &u64) -> u64 {
        let card_dist: Uniform<u8> = Uniform::new(0, 51);
        for _ in 0..1000 {
            let c = card_dist.sample(rng);
            if ((used_cards_mask | self.board_mask) & 1u64 << c) == 0 {
                return self.board_mask | 1u64 << c;
            }
        }
        // too many attempts
        assert!(false);
        return 0;
    }
    // return an array of all public chance outcomes
    pub fn sample_all(&self, used_cards_mask: &u64) -> Vec<u64> {
        let mut boards: Vec<u64> = Vec::new();
        for i in 0..52 {
            if ((used_cards_mask | self.board_mask) & 1u64 << i) == 0 {
                boards.push(self.board_mask | 1u64 << i);
            }

        }
        return boards;
    }
}

pub enum GameTreeNode {
    ActionNode(ActionNode),
    PrivateChanceNode(PrivateChanceNode),
    PublicChanceNode(PublicChanceNode),
    TerminalNode(TerminalNode)
}
