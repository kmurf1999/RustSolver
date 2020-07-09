mod state;
mod arena;
mod nodes;
mod actions;
mod options;
mod tree_builder;


use tree_builder::{TreeBuilder};
use state::{BettingRound, PlayerState, GameState};



fn main() {
    let options = options::default_options();
    let mut tree = TreeBuilder::init(&options);
    tree.build();
    tree.print_node(0, 0);
}
