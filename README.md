## Information Abstraction

 - Preflop: isomorphism
 - Flop: potential-aware
 - Turn: EMD
 - River: OHSC

### Opponent Hand Strength Clustering (OHSC)

First create an EMD based abstraction for opponent preflop hands.
This will buckets for all 169 cannonical opponent hands, into n buckets say 8.

Then, for each hand on the current round, (I'll be using this for the river),
Create a histogram that represents the probability of beating each opponent hand cluster.

Then, using the L2 distance functions and Kmeans with multiple restarts, cluster these river hands into K buckets

### Earth Mover's Distance (EMD)

To compute EMD, compute a histogram for each hand in the current round.
This histogram represents the probability distribution of EHS after the remaining cards are dealt.

Then, using the EMD function as a distance metric and K-means with multiple restarts, cluster these hands into K buckets

### Potential Aware abstraction

Can only be used on one round (typically the flop).

Group hands based on the EMD of histograms that represend the probability distribution of the next chance outcome landing the current hand into a next-round bucket.

# Data Design

Infoset table

for each round and for each player maintain a table of intosets
Traverse the public tree one time and count action nodes for each
player in a round

each public chance outcome (board) has a card table which, for each player map:
player-hand in range (u8, u8) -> hand-index -> cluster-idx -> zero indexed possible hands

each round has a board table, which maps each possible board:
board -> zero indexed possible board

so to get the infoset

infoset_table -> [round, player] -> [board] -> [action node idx] -> [hand-index, num_actions]

(round + player)

 get-infoset
 - round
 - player
 - board
