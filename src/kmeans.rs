use rand::distributions::{Uniform};
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::{DataPoint, Histogram};

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
 * performs kmeans a on initialized centers using a specified distance function
 */
pub fn kmeans(k: usize, dataset: &Vec<DataPoint>,
        dist_func: &dyn Fn(&Histogram, &Histogram) -> f32,
        center: &mut Vec<Histogram>) {
}

/**
 * Computes the L2 distance between two histograms
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
