use crossbeam::atomic::AtomicCell;

use std::collections::HashSet;
use std::sync::Arc;
use std::io::Write;
use std::io;
use std::time::{ Instant };

use rand::distributions::{Distribution, WeightedIndex, Uniform};
use crate::rand::prelude::SliceRandom;

use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::{Histogram};

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

// static EPSILON: f32 = 0.01;

pub struct Kmeans {
    centers: Vec<Histogram>
}

impl Kmeans {
    /// Kmeans ++ initialization
    pub fn init_pp<R: Rng>(
        n_centers: usize, rng: &mut R,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
        dataset: &Vec<Histogram>) -> Self {

        let start = Instant::now();

        println!("Initializing kmeans++ with {} centers", n_centers);

        let n_data = dataset.len();
        // push initial center randomly
        let mut centers: Vec<&Histogram> = Vec::with_capacity(n_centers);
        centers.push(&dataset[rng.gen_range(0, n_data)]);

        let mut min_dists= vec![f32::MAX; n_data];
        for i in 1..n_centers {

            print!("Center: {}/{}\r", i, n_centers);
            io::stdout().flush().unwrap();

            update_min_dists(dist_func, &mut min_dists, dataset, &centers[i-1]);
            let dist = WeightedIndex::new(&min_dists).unwrap();
            centers.push(&dataset[dist.sample(rng)]);
        }

        println!("Done.  Took {}ms", start.elapsed().as_millis());

        Kmeans {
            centers: centers.iter().map(|x| (*x).clone()).collect()
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
            n_restarts: usize, n_centers: usize, rng: &mut R,
            dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
            dataset: &Vec<Histogram>) -> Kmeans {

        let start = Instant::now();

        println!("Initializing Kmeans with {} random restarts", n_restarts);

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
        let iteration = AtomicCell::new(0usize);
        let mut cluster_dists: Vec<f32> = vec![0f32; n_restarts];
        cluster_dists.par_iter_mut().enumerate().for_each(|(r, cd)| {

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

        println!("Done.  Took {}ms", start.elapsed().as_millis());

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
            dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync)) -> f32 {

        if clusters.len() != dataset.len() {
            panic!("Cluster and dataset does not match");
        }

        // number of means
        let n_centers = self.centers.len();

        // number of clusters that have changed
        let inertia = AtomicCell::new(0f32);

        clusters.par_iter_mut().enumerate().for_each(|(i, cluster)| {
            let curr_cluster = *cluster;
            let mut variance = vec![0.0; n_centers];
            let mut min_cluster = 0;
            variance[0] = dist_func(
                &dataset[i],
                &self.centers[0]);
            let mut min_variance = variance[0];
            for k in 1..n_centers {
                variance[k] = dist_func(
                    &dataset[i],
                    &self.centers[k]);
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

    /// train kmeans using random small batches of data
    pub fn fit_minibatch<R: Rng>(&mut self,
        rng: &mut R,
        dataset: &Vec<Histogram>,
        max_iteration: usize,
        batch_size: usize,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
    ) {

        let start = Instant::now();
        let k = self.centers.len();
        let n_bins = dataset[0].len();

        let mut s = vec![f32::MAX; k];
        let mut inertia = vec![0f32; batch_size];
        let mut inertia_sum = 0f32;

        for iteration in 0..max_iteration {

            let iter_start = Instant::now();

            print!("Getting train data\r");
            io::stdout().flush().unwrap();

            let train_data: Vec<Histogram> = dataset
                .choose_multiple(rng, batch_size)
                .cloned()
                .collect();

            let mut bounds = vec![(0f32, f32::MAX); batch_size];
            let mut clusters = vec![0usize; batch_size];

            print!("Calculating s\n");
            io::stdout().flush().unwrap();

            // do one iteration of kmeans
            self.init_s(&mut s, dist_func);

            print!("Assigning Clusters s\n");
            io::stdout().flush().unwrap();

            self.reassign_clusters(
                &train_data, &s,
                &mut clusters, &mut bounds,
                &mut inertia, dist_func
            );

            print!("Calculating new means\n");
            io::stdout().flush().unwrap();

            let mut cluster_elem_counter: Vec<f32> = vec![0.0; k];
            let mut cluster_prob_mass: Vec<Vec<f32>> = vec![vec![0.0; n_bins]; k];
            for j in 0..batch_size {
                cluster_elem_counter[clusters[j]] += 1.0;
                for k in 0..n_bins {
                    cluster_prob_mass[clusters[j]][k] +=
                        dataset[j][k];
                }
            }

            let new_centers: Vec<Vec<f32>> = cluster_prob_mass
                .par_iter_mut()
                .enumerate()
                .map(|(j, cpm)| {
                    // let lr = 1.0 / cluster_elem_counter[j];
                    for k in 0..n_bins {
                        if cpm[k] > 0.0 {
                            cpm[k] /= cluster_elem_counter[j];
                        }
                    }
                    // for k in 0..n_bins {
                    //     cpm[k] = ((1.0 - lr) * self.centers[j][k]) + (lr * cpm[k]);
                    // }
                    cpm.to_owned()
            }).collect();

            // print!("Updating bounds\n");
            // io::stdout().flush().unwrap();

            // get movement of each center
            // let center_movement: Vec<f32> = (0..k).into_par_iter().map(|j| {
            //     dist_func(&new_centers[j], &self.centers[j])
            // }).collect();

            // let mut longest_idx = 0;
            // let mut longest = center_movement[0];
            // let mut second_longest = center_movement[1];
            // if longest < second_longest {
            //     longest = center_movement[1];
            //     second_longest = center_movement[0];
            //     longest_idx = 1;
            // }
            // for j in 2..k {
            //     if longest < center_movement[j] {
            //         second_longest = longest;
            //         longest = center_movement[j];
            //         longest_idx = j;
            //     } else if second_longest < center_movement[j] {
            //         second_longest = center_movement[j];
            //     }
            // }

            // bounds.par_iter_mut().enumerate().for_each(|(i, b)| {
            //     b.1 += center_movement[clusters[i]];
            //     b.0 -= if clusters[i] == longest_idx {
            //         second_longest
            //     } else {
            //         longest
            //     };
            // });

            let next_inertia_sum = inertia.par_iter().sum::<f32>();
            print!("iteration: {}, inertia: {:.4}, ({:.4})\r",
                   iteration,
                   next_inertia_sum,
                   next_inertia_sum - inertia_sum);
            io::stdout().flush().unwrap();
            inertia_sum = next_inertia_sum;

            self.centers = new_centers;
        }
    }

    fn init_s(&self,
        s: &mut Vec<f32>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync)
    ) {
        let k = s.len();
        s.par_iter_mut().enumerate().for_each(|(i, s)| {
            for j in 0..k {
                if i == j { continue; }
                let d = dist_func(
                    &self.centers[i],
                    &self.centers[j]);
                if d < *s {
                    *s = d;
                }
            }
            *s /= 2.0;
        });
    }

    fn reassign_clusters(&mut self,
        dataset: &Vec<Histogram>,
        s: &Vec<f32>,
        clusters: &mut Vec<usize>,
        bounds: &mut Vec<(f32, f32)>,
        inertia: &mut Vec<f32>,
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync)) {

        let k = s.len();
        clusters.par_iter_mut()
            .zip(bounds.par_iter_mut())
            .zip(inertia.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((ci, bi), di))| {

                let mut min_cluster = *ci;
                let upper_comp_bound = max!(s[min_cluster], bi.0);
                if bi.1 <= upper_comp_bound { return; }
                let mut u2 = dist_func(
                    &dataset[i],
                    &self.centers[min_cluster]);
                bi.1 = u2;
                if bi.1 <= upper_comp_bound { return; }
                // update lower bound by looking at all other centers
                let mut l2 = f32::MAX;
                for j in 0..k {
                    if j == min_cluster { continue; }

                    let dist2 = dist_func(
                        &dataset[i],
                        &self.centers[j]);

                    if dist2 < u2 {
                        l2 = u2;
                        u2 = dist2;
                        min_cluster = j;
                        *di = dist2;
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

    /// Fits kmeans to dataset with dist function
    pub fn fit_regular(&mut self,
           dataset: &Vec<Histogram>,
           dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync)
    ) -> (Vec<usize>, f32) {

        let start = Instant::now();
        let k = self.centers.len();
        let n_data = dataset.len();
        let n_bins = dataset[0].len();

        println!("Fitting {} centers to dataset", k);

        let mut iteration: usize = 0;
        // which cluster each item in dataset is in
        let mut clusters: Vec<usize> = vec![0; n_data];
        // s is the distance between a cluster and the closest cluster / 2
        let mut s = vec![f32::MAX; k];
        // distance from datapoint to assigned cluster
        let mut inertia = vec![0f32; n_data];
        // bounds for calculating current cluster
        let mut bounds = vec![(0f32, f32::MAX); n_data];

        loop {
            iteration += 1;
            // calculate s
            self.init_s(&mut s, dist_func);
            self.reassign_clusters(
                dataset, &s,
                &mut clusters, &mut bounds,
                &mut inertia, dist_func);
            // update centers
            // calculate new means
            let mut cluster_elem_counter: Vec<f32> = vec![0.0; k];
            let mut cluster_prob_mass: Vec<Vec<f32>> = vec![vec![0.0; n_bins]; k];
            for j in 0..n_data {
                cluster_elem_counter[clusters[j]] += 1.0;
                for k in 0..n_bins {
                    cluster_prob_mass[clusters[j]][k] +=
                        dataset[j][k];
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

            }).collect();

            // get movement of each center
            let center_movement: Vec<f32> = (0..k).into_par_iter().map(|j| {
                dist_func(&new_centers[j], &self.centers[j])
            }).collect();

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
            if longest < 0.000000001 {
                break;
            } else {
                print!("iteration: {}, inertia: {:.4}\r",
                       iteration,
                       inertia.iter().sum::<f32>());
                io::stdout().flush().unwrap();
            }

            self.centers = new_centers;
        }

        println!("Done.  Took: {}ms, inertia: {}",
            start.elapsed().as_millis(),
            inertia.iter().sum::<f32>());

        return (clusters, inertia.iter().sum());
    }
}

// used for kmeans ++
pub fn update_min_dists(
        dist_func: &'static (dyn Fn(&Histogram, &Histogram) -> f32 + Sync),
        min_dists: &mut Vec<f32>,
        dataset: &Vec<Histogram>,
        new_center: &Histogram) {

    min_dists.par_iter_mut().enumerate().for_each(|(i, min_dist)| {
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
