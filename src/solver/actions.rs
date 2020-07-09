pub static ACTIONS: [Action; 5] = [
    Action::Bet(1.0),
    Action::Raise(2.0),
    Action::Check,
    Action::Call,
    Action::Fold
];

#[derive(Debug, Copy, Clone)]
pub enum Action {
    Bet(f32),
    Raise(f32),
    Check,
    Call,
    Fold
}

impl Action {
    pub fn to_string(&self) -> &str {
        return match self {
            Action::Check => "Check",
            Action::Bet(_) => "Bet",
            Action::Raise(_) => "Raise",
            Action::Fold => "Fold",
            Action::Call => "Call"
        }
    }
}
