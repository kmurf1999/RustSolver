/**
* Functions to approximate EMD of two histograms in linear time
*
* https://www.hindawi.com/journals/mpe/2014/406358/
*/

// use std::cmp;

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

// macro_rules! max {
//     ($x: expr) => ($x);
//     ($x: expr, $($z: expr),+) => {{
//         let y = max!($($z),*);
//         if $x > y {
//             $x
//         } else {
//             y
//         }
//     }}
// }

// fn l2_norm(x: &Vec<i64>) -> f64 {
//     let mut sum: f64 = 0.0;
//     for i in 0..x.len() {
//         sum += (x[i] * x[i]) as f64;
//     }
//     return sum.sqrt();
// }

// fn get_bins(b: Vec<i64>, B: &mut Vec<Vec<i64>>, u: f64, d: usize, curr: usize) {
//     for i in curr..d {
//         let mut bp = b.clone();
//         if bp[i] == 0 {
//             bp[i] -= 1;
//             if l1_norm(&bp) < u {
//                 B.push(bp.clone());
//                 get_bins(bp, B, u, d, i);
//             }
//             bp = b.clone();
//             bp[i] += 1;
//             if l1_norm(&bp) < u {
//                 B.push(bp.clone());
//                 get_bins(bp, B, u, d, i);
//             }
//         } else {
//             if bp[i] < 0 {
//                 bp[i] -= 1;
//             } else {
//                 bp[i] += 1;
//             }
//             if l1_norm(&bp) < u {
//                 B.push(bp.clone());
//                 get_bins(bp, B, u, d, i);
//             }
//         }
//     }
// }

/**
 * recursivily get reachable bins given a distance threshold u
 */
fn get_bins_1d(b: isize, bins: &mut Vec<isize>, u: isize) {
    let mut bp = b;
    if bp == 0 {
        bp -= 1;
        if bp.abs() < u {
            bins.push(bp);
            get_bins_1d(bp, bins, u);
        }
        bp = b;
        bp += 1;
        if bp.abs() < u {
            bins.push(bp);
            get_bins_1d(bp, bins, u);
        }
    } else {
        if bp < 0 {
            bp -= 1;
        } else {
            bp += 1;
        }
        if bp.abs() < u {
            bins.push(bp);
            get_bins_1d(bp, bins, u);
        }
    }
}

pub fn emd_1d(p: &Vec<f64>, q: &Vec<f64>) -> f64 {
    let mut p = p.clone();
    let mut q = q.clone();

    // main computation
    let mut cost = 0.0;
    let mut w = 0.0;

    // corresponding bins (no cost)
    for i in 0..q.len() {
        let mass = min!(p[i], q[i]);
        w += mass;
        p[i] -= mass;
        q[i] -= mass;
    }

    let u: isize = q.len() as isize;
    // let u: isize = if w < 0.5 {
    //     q.len() as isize
    // } else {
    //     (q.len() / 2) as isize
    // };

    let mut b: Vec<isize> = Vec::new();
    get_bins_1d(0, &mut b, u);
    b.sort_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap());

    // cross bin
    for i in 0..b.len() {
        for j in 0..p.len() {
            if p[j] != 0.0 && (j as isize + b[i]) >= 0 {
                let k = (j as isize + b[i]) as usize;
                if k < q.len() && q[k] != 0.0 {
                    let mass = min!(p[j], q[k]);
                    w += mass;
                    cost += mass * (j as f64 - k as f64).abs();
                    p[j] -= mass;
                    q[k] -= mass;
                }
            }
        }
    }

    return cost + (1.0 - w) * u as f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    const ERROR: f64 = 0.05;

    #[test]
    fn test_emd_ak_aq() {
        // histogram with 30 bins
        let ak = vec![
            0.0,0.0,0.0,0.0,0.0025,0.014,0.003,0.018,
            0.0115,0.0215,0.0755,0.0435,0.0135,0.0615,
            0.0855,0.075,0.0125,0.0185,0.0215,0.016,
            0.0085,0.0195,0.0175,0.0235,0.0545,0.0845,
            0.102,0.044,0.035,0.1175
        ];
        let aq = vec![
            0.0,0.0,0.0,0.0,0.001,0.018,0.0035,0.0085,
            0.029,0.0125,0.094,0.0375,0.01,0.057,0.102,
            0.072,0.013,0.0105,0.0185,0.016,0.0115,0.01,
            0.009,0.0085,0.0535,0.069,0.1315,0.05,0.036,0.118
        ];
        let emd = emd_1d(&ak, &aq);
        let actual_value = 0.2835;
        assert!(emd < (actual_value + ERROR));
        assert!(emd > (actual_value - ERROR));
    }

    #[test]
    fn test_emd_66_jt() {
        // 6s6h
        let hist_a = vec![
            0.0,0.0,0.0005,0.0065,0.0025,0.0005,0.0065,0.003,0.0115,
            0.0095,0.0135,0.023,0.012,0.038,0.0705,0.0625,0.0725,0.082,
            0.1005,0.052,0.036,0.036,0.047,0.023,0.025,0.022,0.0355,
            0.035,0.04,0.1335
        ];
        // JsTs
        let hist_b = vec![
            0.0035,0.008,0.0085,0.0205,0.034,0.032,0.007,0.043,0.0875,
            0.0075,0.036,0.0405,0.0175,0.017,0.025,0.036,0.009,0.0095,
            0.0145,0.0245,0.057,0.056,0.055,0.035,0.0395,0.0215,0.042,
            0.042,0.057,0.114
        ];
        let emd = emd_1d(&hist_a, &hist_b);
        let actual_value = 2.709499043500001;
        assert!(emd < (actual_value + ERROR));
        assert!(emd > (actual_value - ERROR));
    }

    #[test]
    fn test_emd_27_aa() {
        let hist_a = vec![
            0.054,0.151,0.0345,0.12,0.014,0.012,0.0095,0.007,0.0135,
            0.018,0.0185,0.0455,0.0835,0.014,0.03,0.05,0.057,0.0395,
            0.018,0.012,0.0185,0.0175,0.0105,0.009,0.01,0.0135,0.046,
            0.0405,0.009,0.024
        ];
        let hist_b = vec![
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0015,0.0015,
            0.0035,0.0,0.0,0.0085,0.0215,0.0065,0.0085,0.0015,0.025,
            0.0385,0.0125,0.026,0.097,0.21,0.1455,0.111,0.082,0.1995
        ];
        let emd = emd_1d(&hist_a, &hist_b);
        let actual_value = 14.220495694500006;
        assert!(emd < (actual_value + ERROR));
        assert!(emd > (actual_value - ERROR));
    }

    #[bench]
    fn bench_emd(b: &mut Bencher) {
        let hist_a = vec![
            0.054,0.151,0.0345,0.12,0.014,0.012,0.0095,0.007,0.0135,
            0.018,0.0185,0.0455,0.0835,0.014,0.03,0.05,0.057,0.0395,
            0.018,0.012,0.0185,0.0175,0.0105,0.009,0.01,0.0135,0.046,
            0.0405,0.009,0.024
        ];
        let hist_b = vec![
            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0015,0.0015,
            0.0035,0.0,0.0,0.0085,0.0215,0.0065,0.0085,0.0015,0.025,
            0.0385,0.0125,0.026,0.097,0.21,0.1455,0.111,0.082,0.1995
        ];
        b.iter(|| emd_1d(&hist_a, &hist_b));
    }
}
