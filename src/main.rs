#![feature(duration_as_u128)]
#![feature(exclusive_range_pattern)]
extern crate rayon;
extern crate byteorder;
extern crate ndarray;
extern crate ndarray_npy;
extern crate itertools;
extern crate num;
#[macro_use] extern crate structopt;

use rayon::prelude::*;
use ndarray::prelude::*;

use num::Zero;
use num::One;
use std::cmp::min;
use std::collections::HashSet;
use std::hash::Hash;
use std::io::prelude::*;
use std::fs;
use std::ops::Add;
use std::path::{Path,PathBuf};
use byteorder::{LittleEndian, WriteBytesExt};
use ndarray_npy::NpzReader;
use ndarray_npy::NpzWriter;
use ndarray_npy::ReadNpzError;
use structopt::StructOpt;

// program argument handeling
#[derive(StructOpt, Debug, Clone)]
#[structopt(name="dister")]
struct Arguments {

    /// input .npz file
    #[structopt(name="INPUT_FILE", short="i", long="input")]
    input: PathBuf,

    /// output directory
    #[structopt(name="OUTPUT_DIRECTORY", short="o", long="output")]
    outdir: PathBuf
}

/// get the unique values on the array
fn get_unique<T: Eq + Hash + Clone>(array: &Array2<T>) -> Vec<T> {
    let mut set: HashSet<T> = HashSet::new();
    for v in array.into_iter() {
        set.insert(v.clone());
    }
    set.into_iter().collect()
}

fn window_min<T: Eq+Ord+Sync+Copy>(data: &Array2<T>, i: usize, j: usize) -> T {
    let window: [T; 8] = unsafe { [
        *data.uget([i+1,j-1]),
        *data.uget([i+1,j]),
        *data.uget([i+1,j+1]),
        *data.uget([i,j-1]),
        *data.uget([i,j+1]),
        *data.uget([i-1,j-1]),
        *data.uget([i-1,j]),
        *data.uget([i-1,j+1]),
    ]};

    min(
        min(min(window[0], window[1]), min(window[2], window[3])),
        min(min(window[4], window[5]), min(window[6], window[7]))
    )
}


/// return true if a value was updated
fn update<T: Eq+Ord+Sync+Copy+Add+One+Zero>(array: &mut Array2<T>, i: usize, j: usize) -> bool {
    unsafe {
        // get the current value
        let current = *array.uget([i,j]);
        // check for the zero condition
        if current == T::zero() { return false; }
        // determine the minimum window value and increment by 1
        let win_min = window_min(&array, i, j) + T::one();
        // update as necessary
        if win_min < current {
            *array.uget_mut([i,j]) = win_min;
            return true;
        }
    }
    // default return
    false
}

fn load<P: AsRef<Path>>(filename: P) -> Result<Array2<i8>, ReadNpzError> {
    let f = fs::File::open(&filename).expect("Cannot create decoder!");;
    let mut decoder = NpzReader::new(f).expect("Cannot create decoder!");
    decoder.by_index(0)
}


fn main() {

    // load options
    let opt = Arguments::from_args();
    println!("{:#?}", opt);

    if !opt.input.exists() {
        eprintln!("input file does not exist!");
        ::std::process::exit(1);
    }

    // create the directories
    if !opt.outdir.exists() {
        fs::create_dir_all(&opt.outdir).expect("Could Not Create Output Directory!");
    }

    // load csv of dispersion distances
    let mut dispersion_table: HashMap<i16, f32> = HashMap::new();

    // load array from file
    let input = load(&opt.input).expect("couldn't load numpy array!");
    let (height, width): (usize,usize) = (input.shape()[0], input.shape()[1]);
    println!("input is size {0} x {1}", height, width);

    let values = get_unique(&input);
    println!("there are {} values.", values.len());

    // compute a grid for each array
    values.par_iter().for_each(|v| {

        // time the iteration
        let now = ::std::time::Instant::now();

        // prepare the array to perform jacobi
        // iteration of array. Clone the input
        // array into a local array. Classify
        // this value, v, on the array as 0
        // distance. Everything else is a maximum
        // distance.
        let mut array: Array2<i16> = Array2::from_elem((height+2, width+2), i16::max_value());
        for i in 0..height {
            for j in 0..width{
                array[[i+1,j+1]] = if input[[i,j]] == *v { 0i16 } else { i16::max_value() - i16::one() };
            }
        }

        // compute manhattan distance values
        // using a jacobi iterator technique.
        let v = *v as i16;

        // determine maximum iterations that will be
        // allowable based on the size of the input data
        let max_iters = (((height as f32).powf(2.) + (width as f32).powf(2.)).sqrt() as usize);
        println!("begin jacobian iteration on ecoclass {}", v);

        // jacobi iteration
        let mut cntr = 0;
        loop {
            let mut do_loop = false;
            for i in 1..height {
                for j in 1..width {
                    do_loop = do_loop | update::<i16>(&mut array, i, j);
                }
            }
            cntr += 1;
            if !do_loop { break; }
            if cntr > max_iters { eprintln!("did not converge!"); break; }
        }

        // helpful messages
        println!("estimated maximum iterations: {}", max_iters);
        println!("convergence reached in {} iterations.", cntr);
        println!("jacobi iteration time: {}ms", now.elapsed().as_millis());

        // extimate euclidean distances using
        // the manhattan distances with an
        // estimated error of 3/pi (-check error)
        output_i16(&array, &opt.outdir.join(format!("out-array-{}.npz", v)));

        // reset the clock
        let now = ::std::time::Instant::now();

        // perform distance estimation
        let mut distance: Array2<f32> = Array2::from_elem((height, width), 0.0);
        for i in 0..height {
            for j in 0..width {
                distance[[i,j]] = (array[[i,j]] as f32) * (::std::f32::consts::PI/4.);
            }
        }

        // helpful message
        println!("distance estimation time: {}ms", now.elapsed().as_millis());

        // produce an output array for the distance
        output_f32(&distance, &opt.outdir.join(format!("distance-{}.npz", v)));

        // apply the dispersion metrics to the distance array
        if let Some(pix_per_yr) = dispersion_table.get(&v) {
            let mut dispersion: Array2<f32> = Array2::from_elem((height, width), 0.0);
            for i in 0..height {
                for j in 0..width {
                    // convert distance units in pixels
                    // to the year since 0 that it will
                    // take to spread the ecoclass to a
                    // specific location on the grid.
                    dispersion[[i,j]] = distance[[i,j]] / pix_per_yr;
                }
            }

            // helpful message
            println!("dispersion computation complete for: {}", v);

            // produce an output distance per year array
            output_f32(&dispersion, &opt.outdir.join(format!("dispersion-{}.npz", v)));
        } else {
            eprintln!("dispersion not computed for: {}", v);
        }

    });

}

fn output_i16<P: AsRef<Path>>(array: &Array2<i16>, path: P) {
    // get data of array
    if let Ok(mut writer) = fs::File::create(path) {
        let mut npz = NpzWriter::new_compressed(writer);
        npz.add_array("a", &array);
    }
}
fn output_f32<P: AsRef<Path>>(array: &Array2<f32>, path: P) {
    // get data of array
    if let Ok(mut writer) = fs::File::create(path) {
        let mut npz = NpzWriter::new_compressed(writer);
        npz.add_array("a", &array);
    }
}