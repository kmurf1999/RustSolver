use crossbeam::atomic::AtomicCell;

use std::cmp::Ordering::Equal;
use std::collections::HashSet;
use std::io;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use crate::rand::prelude::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};

use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::Histogram;

const N_THREADS: usize = 16;

macro_rules! max {
    ($x: expr) => ($x);
    ($x: expr, $($z: expr),+) => {{
        let y = max!($($z),*);
        if $x > y {
            $x
        } else {
            y
        }
    }}
}

macro_rules! min {
    ($x: expr) => ($x);
    ($x: expr, $($z: expr),+) => {{
        let y = min!($($z),*);
        if $x < y {
            $x
        } else {
            y
        }
    }}
}

fn print_vector(arr: &Histogram) {
    for i in 0..arr.len() {
        print!("{:.3} ", arr[i]);
    }
    println!("");
}

// static EPSILON: f32 = 0.01;

pub struct Kmeans {
    centers: Vec<Histogram>,
}

impl Kmeans {
    /// Kmeans ++ initialization
    pub fn init_pp<R: Rng>(
        n_centers: usize,
        rng: &mut R,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
        dataset: &Vec<Histogram>,
    ) -> Self {
        let start = Instant::now();

        println!("Initializing kmeans++ with {} centers", n_centers);

        let n_data = dataset.len();
        // push initial center randomly
        let mut centers: Vec<&Histogram> = Vec::with_capacity(n_centers);
        centers.push(&dataset[rng.gen_range(0, n_data)]);

        let mut min_dists = vec![f32::MAX; n_data];
        for i in 1..n_centers {
            print!("Center: {}/{}\r", i, n_centers);
            io::stdout().flush().unwrap();

            update_min_dists(dist_func, &mut min_dists, dataset, &centers[i - 1]);
            let dist = WeightedIndex::new(&min_dists).unwrap();
            centers.push(&dataset[dist.sample(rng)]);
        }

        println!("Done.  Took {}ms", start.elapsed().as_millis());

        Kmeans {
            centers: centers.iter().map(|x| (*x).clone()).collect(),
        }
    }

    ///  
    ///  Trys n times to initialize the centers for k-means
    ///  randomly chooses centers and return most spread out one
    ///
    ///  # Arguments
    ///  
    ///  * `n_restarts` number of restarts
    ///  * `n_centers` k in k-means
    ///  * `center` k means to return
    ///  * `dataset` reference to dataset
    ///  * `rng` seeded rng
    ///  
    pub fn init_random<R: Rng>(
        n_restarts: usize,
        n_centers: usize,
        rng: &mut R,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
        dataset: &Vec<Histogram>,
    ) -> Kmeans {
        let start = Instant::now();

        println!("Initializing Kmeans with {} random restarts", n_restarts);

        // create n centers to choose from
        let mut centers: Vec<Vec<&Histogram>> = Vec::with_capacity(n_restarts);
        // init centers randomly
        for _ in 0..n_restarts {
            // random init
            centers.push(dataset.choose_multiple(rng, n_centers).collect());
        }

        // calculate total dists of each restart
        let iteration = AtomicCell::new(0usize);
        let mut cluster_dists: Vec<f32> = vec![0f32; n_restarts];
        cluster_dists
            .par_iter_mut()
            .enumerate()
            .for_each(|(r, cd)| {
                let cur_iter = iteration.fetch_add(1);
                print!("Restart: {}/{}\r", cur_iter, n_restarts);
                io::stdout().flush().unwrap();

                let mut sum = 0f32;
                let mut count = 0usize;
                let mut distances = vec![0f32; n_centers];
                for i in 0..n_centers {
                    for j in 0..n_centers {
                        if j == i {
                            continue;
                        }
                        let dist = dist_func(&centers[r][i], &centers[r][j]);
                        distances[i] += dist;
                        count += 1;
                    }
                    sum += distances[i];
                }
                *cd = sum / count as f32;
            });

        // get max index
        // use index of maximum (most spread out) clusters
        let max_cluster: usize = cluster_dists
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();

        println!("Done.  Took {}ms", start.elapsed().as_millis());

        // clone center to return
        Kmeans {
            centers: centers[max_cluster].iter().map(|x| (*x).clone()).collect(),
        }
    }

    /**
     * Fit data to clusters
     * clusters: a mutable reference which contains the predictions
     * returns number of clusters that have changed (used for training)
     */
    pub fn predict(
        &self,
        dataset: &Vec<Histogram>,
        clusters: &mut Vec<usize>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
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

    fn assignment_with_bounds(
        &mut self,
        dataset: &Vec<&Histogram>,
        s: &Vec<f32>,
        clusters: &mut Vec<usize>,
        bounds: &mut Vec<(f32, f32)>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
    ) {
        let k = s.len();
        let clen  = clusters.len();
        println!("{}", clen);
        clusters
            .par_iter_mut()
            .zip(bounds.par_iter_mut())
            .enumerate()
            .for_each(|(i, (ci, bi))| {
                let mut min_cluster = *ci;
                let upper_comp_bound = s[min_cluster].max(bi.0);
                if bi.1 <= upper_comp_bound {
                    return;
                }
                let mut u2 = dist_func(&dataset[i], &self.centers[min_cluster]);
                bi.1 = u2;
                if bi.1 <= upper_comp_bound {
                    return;
                }
                // update lower bound by looking at all other centers
                let mut l2 = f32::MAX;
                for j in 0..k {
                    if j == min_cluster {
                        continue;
                    }

                    let dist2 = dist_func(&dataset[i], &self.centers[j]);

                    if dist2 < u2 {
                        l2 = u2;
                        u2 = dist2;
                        min_cluster = j;
                    } else if dist2 < l2 {
                        l2 = dist2;
                    }
                }
                bi.0 = l2;

                if *ci != min_cluster {
                    bi.1 = u2;
                    *ci = min_cluster;
                }
            });
    }

    fn init_s(
        &self,
        s: &mut Vec<f32>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
    ) {
        let k = s.len();
        s.par_iter_mut().enumerate().for_each(|(i, s)| {
            for j in 0..k {
                if i == j {
                    continue;
                }
                let d = dist_func(&self.centers[i], &self.centers[j]);
                if d < *s {
                    *s = d;
                }
            }
            *s /= 2.0;
        });
    }

    fn reassign_clusters(
        &mut self,
        dataset: &Vec<Histogram>,
        s: &Vec<f32>,
        clusters: &mut Vec<usize>,
        bounds: &mut Vec<(f32, f32)>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
    ) {
        let k = s.len();
        clusters
            .par_iter_mut()
            .zip(bounds.par_iter_mut())
            .enumerate()
            .for_each(|(i, (ci, bi))| {
                let mut min_cluster = *ci;
                let upper_comp_bound = s[min_cluster].max(bi.0);
                if bi.1 <= upper_comp_bound {
                    return;
                }
                let mut u2 = dist_func(&dataset[i], &self.centers[min_cluster]);
                bi.1 = u2;
                if bi.1 <= upper_comp_bound {
                    return;
                }
                // update lower bound by looking at all other centers
                let mut l2 = f32::MAX;
                for j in 0..k {
                    if j == min_cluster {
                        continue;
                    }

                    let dist2 = dist_func(&dataset[i], &self.centers[j]);

                    if dist2 < u2 {
                        l2 = u2;
                        u2 = dist2;
                        min_cluster = j;
                    } else if dist2 < l2 {
                        l2 = dist2;
                    }
                }
                bi.0 = l2;

                if *ci != min_cluster {
                    // update assignment
                    bi.1 = u2;
                    *ci = min_cluster;
                }
            });
    }

    pub fn fit_growbatch<R: Rng>(
        &mut self,
        rng: &mut R,
        dist_func: &'static (dyn Fn(&Vec<f32>, &Vec<f32>) -> f32 + Sync),
        initial_batch_size: usize,
        dataset: &Vec<Vec<f32>>,
    ) {
        println!("Starting growbatch kmeans");
        let threshold = 0.1;
        let stop_threshold = 10000000.0;
        let start = Instant::now();
        let k = self.centers.len();
        let n_bins = self.centers[0].len();
        let n_data = dataset.len();
        let mut shuffled_data: Vec<&Vec<f32>> = dataset.iter().collect();
        shuffled_data.shuffle(rng);
        // non growing arrays
        let mut center_movements: Vec<f32>;
        let mut center_counts = vec![0f32; k];
        let mut center_sums = vec![vec![0f32; n_bins]; k];
        let mut square_dist_sum = vec![0f32; k];
        // s is the dist between a cluster and the nearest cluster / 2
        let mut s = vec![f32::MAX; k];
        // growing arrays
        let mut center_assignments: Vec<usize> = Vec::new();
        let mut bounds: Vec<(f32, f32)> = Vec::new();

        // loop till convergence
        let mut t = 0;
        let mut last_batch_idx = 0;
        let mut current_batch_idx = initial_batch_size;
        loop {
            // update iter-center distances
            self.init_s(&mut s, dist_func);
            // grow arrays
            for _ in last_batch_idx..current_batch_idx {
                bounds.push((0f32, f32::MAX));
                center_assignments.push(0);
            }
            // remove accumulation of old clusters
            for i in 0..k {
                square_dist_sum[i] = 0.0;
                center_counts[i] = 0.0;
                center_sums[i] = vec![0.0; n_bins];
            }
            // assignment with bounds
            self.assignment_with_bounds(
                &shuffled_data,
                &s,
                &mut center_assignments,
                &mut bounds,
                dist_func,
            );
            println!("DONE2");
            for i in 0..100 {
                print!("{} ", center_assignments[i]);
            }
            println!("");
            // accumulate all assignments
            for i in 0..current_batch_idx {
                let new_a = center_assignments[i];
                square_dist_sum[new_a] += bounds[i].1.powf(2.0);
                center_counts[new_a] += 1.0;
                for j in 0..n_bins {
                    center_sums[new_a][j] += shuffled_data[i][j];
                }
            }
            // calculate new centers
            let new_centers: Vec<Vec<f32>> = center_sums
                .iter_mut()
                .enumerate()
                .map(|(i, cs)| {
                    for j in 0..n_bins {
                        if cs[j] > 0.0 && center_counts[i] > 0.0 {
                            cs[j] /= center_counts[i];
                        }
                    }
                    cs.clone()
                })
                .collect();

            // calculate center movements
            center_movements = (0..k)
                .into_par_iter()
                .map(|i| dist_func(&new_centers[i], &self.centers[i]))
                .collect();

            // update bounds
            let mut longest_idx = 0;
            let mut longest = center_movements[0];
            let mut second_longest = center_movements[1];
            if longest < second_longest {
                longest = center_movements[1];
                second_longest = center_movements[0];
                longest_idx = 1;
            }
            for i in 2..k {
                if longest < center_movements[i] {
                    second_longest = longest;
                    longest = center_movements[i];
                    longest_idx = i;
                } else if second_longest < center_movements[i] {
                    second_longest = center_movements[i];
                }
            }
            bounds.par_iter_mut().enumerate().for_each(|(i, b)| {
                b.1 += center_movements[center_assignments[i]];
                b.0 -= if center_assignments[i] == longest_idx {
                    second_longest
                } else {
                    longest
                };
            });
            // get cluster std deviations
            let std_dev: Vec<f32> = (0..k)
                .into_par_iter()
                .map(|i| {
                    if center_counts[i] <= 1.0 {
                        f32::INFINITY
                    } else {
                        (square_dist_sum[i] / (center_counts[i] * (center_counts[i] - 1.0)))
                            .abs()
                            .sqrt()
                    }
                })
                .collect();

            let min_change = (0..k)
                .into_par_iter()
                .map(|i| std_dev[i] / (center_movements[i] + 1e-9))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            self.centers = new_centers;
            last_batch_idx = current_batch_idx;
            // if min_change > threshold {
            current_batch_idx = min!(n_data, current_batch_idx * 2);
            // }
            let inertia = bounds.iter().map(|b| b.1).sum::<f32>() / current_batch_idx as f32;
            if min_change > stop_threshold {
                println!(
                    "Done.  took {}ms, batch size: {}, p: {:.3}, inertia: {:4}",
                    start.elapsed().as_millis(),
                    current_batch_idx,
                    min_change,
                    inertia
                );
                break;
            } else {
                print!(
                    "iteration: {}, batch_size: {}, p: {:.3}, inertia: {:.4}\r",
                    t, current_batch_idx, min_change, inertia
                );
                io::stdout().flush().unwrap();
            }
            t += 1;
            break;
        }
    }

    /// Fits kmeans to dataset with dist function
    pub fn fit_regular(
        &mut self,
        dataset: &Vec<Histogram>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
    ) -> Vec<usize> {
        let start = Instant::now();
        let k = self.centers.len();
        let n_data = dataset.len();
        let n_bins = dataset[0].len();

        println!("Fitting {} centers to dataset", k);

        let mut t: usize = 0;
        // which cluster each item in dataset is in
        let mut clusters: Vec<usize> = vec![0; n_data];
        // s is the distance between a cluster and the closest cluster / 2
        let mut s = vec![f32::MAX; k];
        // bounds for calculating current cluster
        let mut bounds = vec![(0f32, f32::MAX); n_data];

        loop {
            // calculate s
            self.init_s(&mut s, dist_func);
            self.reassign_clusters(dataset, &s, &mut clusters, &mut bounds, dist_func);
            // update centers
            // calculate new means
            let mut cluster_elem_counter: Vec<f32> = vec![0.0; k];
            let mut cluster_prob_mass: Vec<Vec<f32>> = vec![vec![0.0; n_bins]; k];
            for j in 0..n_data {
                cluster_elem_counter[clusters[j]] += 1.0;
                for k in 0..n_bins {
                    cluster_prob_mass[clusters[j]][k] += dataset[j][k];
                }
            }
            let new_centers: Vec<Vec<f32>> = cluster_prob_mass
                .par_iter_mut()
                .enumerate()
                .map(|(j, cbm)| {
                    // calculate mean
                    for k in 0..n_bins {
                        if cbm[k] > 0.0 {
                            cbm[k] /= cluster_elem_counter[j];
                        }
                    }
                    cbm.to_owned()
                })
                .collect();

            // get movement of each center
            let center_movement: Vec<f32> = (0..k)
                .into_par_iter()
                .map(|j| dist_func(&new_centers[j], &self.centers[j]))
                .collect();

            let mut longest_idx = 0;
            let mut longest = center_movement[0];
            let mut second_longest = center_movement[1];
            if longest < second_longest {
                longest = center_movement[1];
                second_longest = center_movement[0];
                longest_idx = 1;
            }
            for j in 2..k {
                if longest < center_movement[j] {
                    second_longest = longest;
                    longest = center_movement[j];
                    longest_idx = j;
                } else if second_longest < center_movement[j] {
                    second_longest = center_movement[j];
                }
            }

            bounds.par_iter_mut().enumerate().for_each(|(i, b)| {
                b.1 += center_movement[clusters[i]];
                b.0 -= if clusters[i] == longest_idx {
                    second_longest
                } else {
                    longest
                };
            });

            // stop if means stop moving
            // pretty much zero
            let inertia = bounds.iter().map(|b| b.1).sum::<f32>() / n_data as f32;
            print!("iteration: {}, inertia: {:.4}\r", t, inertia);
            io::stdout().flush().unwrap();

            self.centers = new_centers;
            t += 1;
            if t == 10 {
                break;
            }
        }

        let inertia = bounds.iter().map(|b| b.1).sum::<f32>() / n_data as f32;
        println!(
            "Done.  Took: {}ms, inertia: {}",
            start.elapsed().as_millis(),
            inertia
        );

        return clusters;
    }
}

// used for kmeans ++
pub fn update_min_dists(
    dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
    min_dists: &mut Vec<f32>,
    dataset: &Vec<Histogram>,
    new_center: &Histogram,
) {
    min_dists
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, min_dist)| {
            let mut d = dist_func(&dataset[i], new_center);
            d = d * d;
            if d < *min_dist {
                *min_dist = d;
            }
        });
}

/// Computes the L2 norm distance between two histograms
pub fn l2_dist(a: &Histogram, b: &Histogram) -> f32 {
    let mut sum = 0f32;
    let mut p_sum: f32;
    for i in 0..a.len() {
        p_sum = a[i] - b[i];
        sum += p_sum * p_sum;
    }
    return sum.sqrt();
}
