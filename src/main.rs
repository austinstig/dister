#![feature(duration_as_u128)]
#![feature(exclusive_range_pattern)]
extern crate rayon;
extern crate ndarray;
extern crate itertools;

use std::cmp::max;
use std::collections::BTreeMap;
use std::collections::HashSet;
use rayon::prelude::*;
use ndarray::prelude::*;
use itertools::Itertools;
use std::hash::Hash;

mod compute;
use self::compute::*;

fn main() {
    let now = ::std::time::Instant::now();

    // setup distance table
    let table = DistanceTable::new(10, 10);

    println!("time-1: {:?}", now.elapsed().as_nanos());
    println!("time-1: {:?}", now.elapsed().as_secs());

    let now = ::std::time::Instant::now();

    println!("{}", table.distance((9,3), (2,7)));

    println!("time-2: {:?}", now.elapsed().as_nanos());
    println!("time-2: {:?}", now.elapsed().as_secs());

    let now = ::std::time::Instant::now();

    let mut array: Array2<i32>  = Array2::from_elem((10000, 10000), -1);
    for i in 0..10000 {
        for j in 0..10000 {
            if i % 3 == 0 {
                array[[i,j]] = 1;
            }
            if j % 7 == 0 {
                array[[i,j]] = 1;
            }
        }
    }

    println!("uniques: {:?}", get_unique(&array));
    let edges = get_edge_points::<i32>(array, 1);
    println!("time-3: {:?}", now.elapsed().as_nanos());
    println!("time-3: {:?}", now.elapsed().as_secs());
    println!("edges: {:?}", edges.len());
}
