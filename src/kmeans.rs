use crossbeam::atomic::AtomicCell;
use std::sync::Arc;
use rand::distributions::{Uniform};
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::{DataPoint, Histogram};

static EPSILON: f32 = 0.01;
static N_THREADS: usize = 8;

/**
 * Trys n times to initialize the centers for k-means
 * returns the most spread out center
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
        dataset: &Vec<DataPoint>, rng: &mut R) {

    // create n centers to choose from
    let mut center_c: Vec<Vec<&Histogram>> = Vec::with_capacity(n_restarts);

    let n_data = dataset.len();
    let dist: Uniform<usize> = Uniform::from(0..n_data);

    // init centers randomly
    for i in 0..n_restarts {
        center_c.push(Vec::with_capacity(n_centers));
        for _ in 0..n_centers {
            center_c[i].push(&dataset[rng.sample(dist)].histogram);
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
                let dist = l2_dist(&center_c[r][i], &center_c[r][j]);
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
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
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
                let mut variance: Vec<f32>;
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
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
        center: &mut Vec<Histogram>) {

    // length of data set
    let n_data: usize = dataset.len();
    // number of features per item in dataset
    let n_bins: usize = dataset[0].histogram.len();

    loop {

        let changed = cluster_data(k, dataset,
                dist_func, center);

        let mut cluster_elem_counter: Vec<f32> = vec![0.0; k];
        let mut cluster_prob_mass: Vec<Vec<f32>> = vec![vec![0.0; n_bins]; k];

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

        println!("{}", changed);

        if (changed as f32 / n_data as f32) <= EPSILON {
            break;
        }
    }
}

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
