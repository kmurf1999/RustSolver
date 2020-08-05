use crossbeam::atomic::AtomicCell;
use hashbrown::HashMap;
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::prelude::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::mem;
use std::ops::AddAssign;
use std::ops::Div;

/// Computes the L2 norm distance between two Vec<f32>s
pub fn l2_dist(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let mut sum = 0f32;
    let mut p_sum: f32;
    for i in 0..a.len() {
        p_sum = a[i] - b[i];
        sum += p_sum * p_sum;
    }
    return sum.sqrt();
}

fn vector_add(a: &mut Vec<f32>, b: &Vec<f32>) {
    let dim = a.len();
    for i in 0..dim {
        a[i] += b[i];
    }
}
fn vector_sub(a: &mut Vec<f32>, b: &Vec<f32>) {
    let dim = a.len();
    for i in 0..dim {
        a[i] -= b[i];
    }
}
fn vector_div(a: &Vec<f32>, b: f32) -> Vec<f32> {
    let mut ret = a.clone();
    if b == 0.0 {
        return ret;
    }
    for i in 0..a.len() {
        ret[i] /= b;
    }
    return ret;
}

// // Vector divide by constant
// impl Div<T> for Vec<T> {
//     // The division of rational numbers is a closed operation.
//     type Output = Self;

//     fn div(self, rhs: T) -> Self::Output {
//         if rhs == 0 {
//             panic!("Cannot divide by zero-valued `Rational`!");
//         }
//         return self.value.iter().map(|v| v / rhs.value).collect();
//     }
// }

// impl AddAssign for Vec<T> {
//     fn add_assign(&mut self, other: Self) {
//         let dim = self.len();
//         if other.len() != dim {
//             panic!("dimensions do not match");
//         }
//         for i in 0..dim {
//             self[i] += other[i];
//         }
//     }
// }

pub struct Kmeans {
    centers: Vec<Vec<f32>>,
}

impl Kmeans {
    /// Kmeans++ initialization
    pub fn init_pp<R: Rng>(
        k: usize,
        rng: &mut R,
        dist_func: &'static (dyn Fn(&Vec<f32>, &Vec<f32>) -> f32 + Sync),
        dataset: &Vec<Vec<f32>>,
    ) -> Self {
        let n_data = dataset.len();
        let mut centers: Vec<Vec<f32>> = Vec::with_capacity(k);
        // push first center randomly
        centers.push(dataset[rng.gen_range(0, n_data)].clone());

        let mut best_sq_distances = vec![f32::MAX; n_data];

        for i in 0..n_data {
            let dist = dist_func(&dataset[i], &centers[0]);
            best_sq_distances[i] = dist * dist;
        }
        for i in 1..k {
            for j in 0..n_data {
                let dist = dist_func(&dataset[j], &centers[i - 1]);
                let sq_dist = dist * dist;
                if sq_dist > best_sq_distances[j] {
                    best_sq_distances[j] = sq_dist;
                }
            }
            let weights = WeightedIndex::new(&best_sq_distances).unwrap();
            centers.push(dataset[weights.sample(rng)].clone());
        }

        Self { centers }
    }

    pub fn growbatch_rho<R: Rng>(
        &mut self,
        rng: &mut R,
        dist_func: &'static (dyn Fn(&Vec<f32>, &Vec<f32>) -> f32 + Sync),
        initial_batch_size: usize,
        dataset: &Vec<Vec<f32>>,
    ) {
        let threshold = 0.0;
        let k = self.centers.len();
        let n_bins = self.centers[0].len();
        let n_data = dataset.len();
        // what cluster idx each datapoint is assigned to
        let mut center_assignments: Vec<usize> = Vec::new();
        // the distance between each cluster and datapoint
        let mut lower: Vec<Vec<f32>> = Vec::new();
        // upper is the distance between a point and its nearest cluster
        let mut upper: Vec<f32> = Vec::new();
        let mut p = vec![0f32; k];
        let mut square_dist_sum = vec![0f32; k];
        let mut cluster_count = vec![0f32; k];
        let mut cluster_sum: Vec<Vec<f32>> = vec![vec![0f32; n_bins]; k];
        let mut sigma_hat = vec![0f32; k];
        // create shuffled data
        let mut shuffled_data: Vec<&Vec<f32>> = dataset.iter().collect();
        shuffled_data.shuffle(rng);
        // loop till convergence
        let mut last_batch_idx = 0;
        let mut current_batch_idx = initial_batch_size;

        let mut t = 0;
        loop {
            for i in last_batch_idx..current_batch_idx {
                lower.push(vec![0f32; k]);
                upper.push(0f32);
                center_assignments.push(0);
            }
            // update old clusters
            for i in 0..last_batch_idx {
                for j in 0..k {
                    lower[i][j] -= p[j];
                }
            }
            for i in 0..last_batch_idx {
                let old_a = center_assignments[i];
                // remove expired sse, S, and v contributions
                square_dist_sum[old_a] -= upper[i].powf(2.0);
                cluster_count[old_a] -= 1.0;
                vector_sub(&mut cluster_sum[old_a], &shuffled_data[i]);
                // assignment with bounds
                for j in 0..k {
                    if center_assignments[i] == j {
                        continue;
                    }
                    if lower[i][j] < upper[i] {
                        lower[i][j] = dist_func(&shuffled_data[i], &self.centers[j]);
                        if lower[i][j] < upper[i] {
                            center_assignments[i] = j;
                            upper[i] = lower[i][j];
                        }
                    }
                }
                // accumulate
                vector_add(&mut cluster_sum[center_assignments[i]], shuffled_data[i]);
                cluster_count[center_assignments[i]] += 1.0;
                square_dist_sum[center_assignments[i]] += upper[i].powf(2.0);
            }
            // setup bounds and assignments for new data
            for i in last_batch_idx..current_batch_idx {
                for j in 0..k {
                    lower[i][j] = dist_func(&shuffled_data[i], &self.centers[j]);
                }
            }
            for i in last_batch_idx..current_batch_idx {
                let mut min_dist = lower[i][0];
                for j in 1..k {
                    if lower[i][j] < min_dist {
                        upper[i] = lower[i][j];
                        center_assignments[i] = j;
                    }
                }

                vector_add(&mut cluster_sum[center_assignments[i]], shuffled_data[i]);
                cluster_count[center_assignments[i]] += 1.0;
                square_dist_sum[center_assignments[i]] += upper[i].powf(2.0);
            }
            // calculate new centers and p values
            for i in 0..k {
                sigma_hat[i] =
                    (square_dist_sum[i] / (cluster_count[i] * (cluster_count[i] - 1.0))).sqrt();
                let old_center = self.centers[i].clone();
                self.centers[i] = vector_div(&cluster_sum[i], cluster_count[i]);
                p[i] = dist_func(&old_center, &self.centers[i]);
            }
            // grow batch size
            last_batch_idx = current_batch_idx;
            let mut min_change = f32::MAX;
            for i in 0..k {
                let change = sigma_hat[i] / p[i];
                if change < min_change {
                    min_change = change;
                }
            }
            println!(
                "min change: {} avg dist: {}",
                min_change,
                sigma_hat.iter().sum::<f32>() / current_batch_idx as f32
            );
            if min_change > threshold {
                current_batch_idx *= 2;
            }
            if current_batch_idx > n_data {
                current_batch_idx = n_data;
            }
            t += 1;
            if t == 25 {
                break;
            }
        }
    }

    /**
     * Fit data to clusters
     * clusters: a mutable reference which contains the predictions
     * returns number of clusters that have changed (used for training)
     */
    pub fn predict(
        &self,
        dataset: &Vec<Vec<f32>>,
        clusters: &mut Vec<usize>,
        dist_func: &'static (dyn Fn(&Vec<f32>, &Vec<f32>) -> f32 + Sync),
    ) -> f32 {
        if clusters.len() != dataset.len() {
            panic!("Cluster and dataset does not match");
        }

        // number of means
        let n_centers = self.centers.len();

        // number of clusters that have changed
        let inertia = AtomicCell::new(0f32);

        clusters
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, cluster)| {
                let curr_cluster = *cluster;
                let mut variance = vec![0.0; n_centers];
                let mut min_cluster = 0;
                variance[0] = dist_func(&dataset[i], &self.centers[0]);
                let mut min_variance = variance[0];
                for k in 1..n_centers {
                    variance[k] = dist_func(&dataset[i], &self.centers[k]);
                    if variance[k] < min_variance {
                        min_variance = variance[k];
                        min_cluster = k;
                    }
                }

                inertia.store(inertia.load() + variance[min_cluster]);

                *cluster = min_cluster;
            });

        return inertia.load();
    }
}
