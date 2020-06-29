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

