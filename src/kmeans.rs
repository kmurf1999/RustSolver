use crossbeam::atomic::AtomicCell;
use std::sync::Arc;
use rand::distributions::{WeightedIndex, Uniform};
use std::io::Write; // <--- ring flush() into scope
use std::io;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::{DataPoint, Histogram};

static EPSILON: f64 = 0.005;
static N_THREADS: usize = 8;

pub fn update_min_dists(
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f64 + Sync),
        min_dists: &mut Vec<f64>,
        dataset: &Vec<DataPoint>,
        new_center: &Histogram) {

    let mut dist;
    for i in 0..dataset.len() {
        dist = dist_func(&dataset[i].histogram, new_center);
        dist = dist * dist;
        if dist < min_dists[i] {
            min_dists[i] = dist;
        }
    }
}

/**
 * Trys n times to initialize the centers for k-means
 * uses Kmeans++ to choose centers
 *
 * n_restarts: number of restarts,
 * n_centers: k in k-means,
 * center: k means to return,
 * dataset: reference to dataset,
 * rng: seeded rng
 */
pub fn kmeans_center_multiple_restarts<R: Rng>(
        n_restarts: usize, n_centers: usize,
        center: &mut Vec<Histogram>,
        dataset: &Vec<DataPoint>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f64 + Sync),
        rng: &mut R) {

    // create n centers to choose from
    let mut center_c: Vec<Vec<&Histogram>> = Vec::with_capacity(n_restarts);

    let n_data = dataset.len();
    // for picking first center
    let uniform_dist: Uniform<usize> = Uniform::from(0..n_data);
    // uses min dists for weighting
    let mut weighted_dist: WeightedIndex<f64>;
    let mut min_dists = vec![f64::MAX; n_data];

    // init centers randomly
    for i in 0..n_restarts {
        center_c.push(Vec::with_capacity(n_centers));
        // push first center
        center_c[i].push(&dataset[rng.sample(uniform_dist)].histogram);
        for k in 1..n_centers {
            // update min dists of datapoints from center
            update_min_dists(dist_func, &mut min_dists, &dataset, &center_c[i][k-1]);
            weighted_dist = WeightedIndex::new(&min_dists).unwrap();
            // selected next center based on x^2 weighted dist
            center_c[i].push(&dataset[rng.sample(weighted_dist)].histogram);
        }

        // update progress
        print!("{}/{}\r", i, n_restarts);
        io::stdout().flush().unwrap();
    }

    // calculate total dists of each restart
    let mut cluster_dists: Vec<f64> = vec![0f64; n_restarts];
    cluster_dists.par_iter_mut().enumerate().for_each(|(r, cd)| {
        let mut sum = 0f64;
        let mut count = 0usize;
        let mut distances = vec![0f64; n_centers];
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
        *cd = sum / count as f64;
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
    *center = center_c[max_cluster].iter().map(|x| (*x).clone()).collect();
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
pub fn cluster_data(k: usize, dataset: &mut Vec<DataPoint>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f64 + Sync),
        center: &Vec<Histogram>) -> usize {

    // length of data
    let n_data = dataset.len();

    // number of clusters that have changed
    let changed = Arc::new(AtomicCell::new(0usize));

    let size_per_thread = n_data / N_THREADS;

    let center = Arc::new(center);

    crossbeam::scope(|scope| {
        for slice in dataset.chunks_mut(size_per_thread) {
            let center = Arc::clone(&center);
            let changed = Arc::clone(&changed);
            scope.spawn(move |_| {
                let mut curr_cluster: usize;
                let mut min_cluster: usize;
                let mut variance: Vec<f64>;
                for i in 0..slice.len() {
                    curr_cluster = slice[i].cluster;
                    variance = vec![0.0; k];

                    for j in 0..k {
                        variance[j] = dist_func(&slice[i].histogram, &center[j]);
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

                    slice[i].cluster = min_cluster;
                }
            });
        }
    }).unwrap();

    return changed.load();
}

/**
 * performs kmeans a on initialized centers using a specified distance function
 */
pub fn kmeans(k: usize, dataset: &mut Vec<DataPoint>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f64 + Sync),
        center: &mut Vec<Histogram>) {

    // length of data set
    let n_data: usize = dataset.len();
    // number of features per item in dataset
    let n_bins: usize = dataset[0].histogram.len();

    let mut iteration: usize = 0;
    let mut accuracy: f64;

    loop {

        let changed = cluster_data(k, dataset,
                dist_func, center);

        let mut cluster_elem_counter: Vec<f64> = vec![0.0; k];
        let mut cluster_prob_mass: Vec<Vec<f64>> = vec![vec![0.0; n_bins]; k];

        for i in 0..n_data {
            cluster_elem_counter[dataset[i].cluster] += 1.0;
            for j in 0..n_bins {
                cluster_prob_mass[dataset[i].cluster][j] +=
                    dataset[i].histogram[j];
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
            center[i] = cluster_prob_mass[i].to_owned();
        }

        // print progress to console
        accuracy = changed as f64 / n_data as f64;
        print!("iteration {}, epsilon: {:.3}\r", iteration, accuracy);
        io::stdout().flush().unwrap();
        iteration += 1;
        if (accuracy) <= EPSILON {
            break;
        }
    }
}

/**
 * Computes the L2 norm distance between two histograms
 */
pub fn l2_dist(a: &Histogram, b: &Histogram) -> f64 {
    let mut sum = 0f64;
    let mut p_sum: f64;
    for i in 0..a.len() {
        p_sum = a[i] - b[i];
        sum += p_sum * p_sum;
    }
    return sum.sqrt();
}
