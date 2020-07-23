use crossbeam::atomic::AtomicCell;

use std::sync::Arc;
use std::io::Write;
use std::io;

use rand::distributions::{WeightedIndex, Uniform};

use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::{Histogram};

static EPSILON: f32 = 0.005;
static N_THREADS: usize = 8;

pub struct Kmeans {
    centers: Vec<Histogram>
}

impl Kmeans {
    /**
     * Trys n times to initialize the centers for k-means
     * randomly chooses centers and return max distance one
     *
     * n_restarts: number of restarts,
     * n_centers: k in k-means,
     * center: k means to return,
     * dataset: reference to dataset,
     * rng: seeded rng
     */
    pub fn init_random<R: Rng>(
            n_restarts: usize, n_centers: usize, rng: &mut R,
            dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
            dataset: &Vec<Histogram>) -> Kmeans {

        println!("Initializing Kmeans++ with {} restarts", n_restarts);

        // create n centers to choose from
        let mut center_c: Vec<Vec<&Histogram>> = Vec::with_capacity(n_restarts);
        let n_data = dataset.len();
        // for picking first center
        let uniform_dist: Uniform<usize> = Uniform::from(0..n_data);
        // init centers randomly
        for i in 0..n_restarts {
            // random init
            center_c.push(Vec::with_capacity(n_centers));
            for _ in 0..n_centers {
             center_c[i].push(&dataset[rng.sample(uniform_dist)]);
            }
        }

        // calculate total dists of each restart
        let mut cluster_dists: Vec<f32> = vec![0f32; n_restarts];
        cluster_dists.par_iter_mut().enumerate().for_each(|(r, cd)| {
            let mut sum = 0f32;
            let mut count = 0usize;
            let mut distances = vec![0f32; n_centers];
            for i in 0..n_centers {
                for j in 0..n_centers {
                    if j == i {
                        continue;
                    }
                    let dist = dist_func(&center_c[r][i], &center_c[r][j]);
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

        // clone center to return
        Kmeans {
            centers: center_c[max_cluster].iter().map(|x| (*x).clone()).collect()
        }
    }

    /**
     * Fit data to clusters
     * clusters: a mutable reference which contains the predictions
     * returns number of clusters that have changed (used for training)
     */
    pub fn predict(&self,
            dataset: &Vec<Histogram>,
            clusters: &mut Vec<usize>,
            dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync)) -> usize {

        // number of means
        let n_centers = self.centers.len();
        // length of data
        let n_data = dataset.len();
        // number of clusters that have changed
        let changed = Arc::new(AtomicCell::new(0usize));
        // let centers = Arc::new(self.centers);

        let size_per_thread = n_data / N_THREADS;

        crossbeam::scope(|scope| {
            for (i, slice) in clusters.chunks_mut(size_per_thread).enumerate() {
                // let center = Arc::clone(&center);
                let changed = Arc::clone(&changed);
                scope.spawn(move |_| {
                    let mut curr_cluster: usize;
                    let mut min_cluster: usize;
                    let mut variance: Vec<f32>;
                    for j in 0..slice.len() {
                        curr_cluster = slice[j];
                        variance = vec![0.0; n_centers];

                        for k in 0..n_centers {
                            variance[k] = dist_func(
                                &dataset[(i * size_per_thread) + j],
                                &self.centers[k]);
                        }

                        // get index of closest mean
                        min_cluster = variance
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                            .map(|(i, _)| i)
                            .unwrap();

                        if min_cluster != curr_cluster {
                            changed.fetch_add(1);
                        }

                        slice[j] = min_cluster;
                    }
                });
            }
        }).unwrap();

        return changed.load();
    }

    pub fn fit(&mut self, dataset: &Vec<Histogram>,
            dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync)
            ) -> Vec<usize> {

        // number of means
        let k = self.centers.len();
        // length of data set
        let n_data: usize = dataset.len();
        // number of features per item in dataset
        let n_bins: usize = dataset[0].len();

        let mut iteration: usize = 0;
        let mut accuracy: f32;

        // which cluster each item in dataset is in
        let mut clusters: Vec<usize> = vec![0; n_data];

        loop {

            let changed = self.predict(dataset, &mut clusters, dist_func);

            let mut cluster_elem_counter: Vec<f32> = vec![0.0; k];
            let mut cluster_prob_mass: Vec<Vec<f32>> = vec![vec![0.0; n_bins]; k];

            for i in 0..n_data {
                cluster_elem_counter[clusters[i]] += 1.0;
                for j in 0..n_bins {
                    cluster_prob_mass[clusters[i]][j] +=
                        dataset[i][j];
                }
            }

            // update centers
            for i in 0..k {
                for j in 0..n_bins {
                    if cluster_prob_mass[i][j] > 0.0 {
                        cluster_prob_mass[i][j] /=
                            cluster_elem_counter[i];
                    }
                }
                self.centers[i] = cluster_prob_mass[i].to_owned();
            }

            // print progress to console
            accuracy = changed as f32 / n_data as f32;
            print!("iteration {}, epsilon: {:.3}\r", iteration, accuracy);
            io::stdout().flush().unwrap();
            iteration += 1;
            if (accuracy) <= EPSILON {
                break;
            }
        }

        return clusters;
    }
}

pub fn update_min_dists(
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
        min_dists: &mut Vec<f32>,
        dataset: &Vec<Histogram>,
        new_center: &Histogram) {

    let mut dist;
    for i in 0..dataset.len() {
        dist = dist_func(&dataset[i], new_center);
        dist = dist * dist;
        if dist < min_dists[i] {
            min_dists[i] = dist;
        }
    }
}

/**
 * Updates the clusters for a dataset based on a given distance function
 * returns the number of elements that have changed
 *
 * k: number of clusters
 * dataset: mutabe reference to list of data to cluster
 * dist_func: distance function to compare histograms
 * center: k-centers
 */

/**
 * performs kmeans a on initialized centers using a specified distance function
 */

/**
 * Computes the L2 norm distance between two histograms
 */
pub fn l2_dist(a: &Histogram, b: &Histogram) -> f32 {
    let mut sum = 0f32;
    let mut p_sum: f32;
    for i in 0..a.len() {
        p_sum = a[i] - b[i];
        sum += p_sum * p_sum;
    }
    return sum.sqrt();
}
