#![feature(duration_as_u128)]
#![feature(try_trait)]
#![feature(exclusive_range_pattern)]
#[macro_use] extern crate failure;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate structopt;
#[macro_use] extern crate derive_more;

// preludes
use std::io::prelude::*;
use rayon::prelude::*;
use ndarray::prelude::*;

// imports
use rand::SeedableRng;
use rand::seq::SliceRandom;
use num::Zero;
use num::One;
use std::cmp::min;
use std::collections::HashSet;
use std::hash::Hash;
use std::fs;
use std::ops::Add;
use std::str::FromStr;
use std::path::{Path,PathBuf};
use std::collections::HashMap;
use ndarray_npy::NpzReader;
use ndarray_npy::NpzWriter;
use ndarray_npy::ReadNpzError;
use structopt::StructOpt;
use failure::Fail;

// *** Type Aliases *** //
type EcoClass = i16;
type Distance = f32;
type Temp = f32;
type Precip = f32;
type Year = u16;
type PixelPerYear = Distance;

/// # Errors
/// errors within the program
#[derive(Debug,Fail,From)]
pub enum Failure {

    // none error
    #[fail(display="none error")]
    NoneError(::std::option::NoneError),

    // parse int error
    #[fail(display="parse int failure")]
    ParseIntError(::std::num::ParseIntError),

    // io error
    #[fail(display="io failure")]
    IOError(::std::io::Error),

    // csv error
    #[fail(display="csv failure")]
    CSVError(csv::Error),

    // npy error
    #[fail(display="npz read failure")]
    NPYReadError(ndarray_npy::ReadNpzError),

    // npy error
    #[fail(display="npz write failure")]
    NPYWriteError(ndarray_npy::WriteNpzError),

    // missing eco class error
    #[fail(display="missing ecological class in dispersion table: {}", value)]
    MissingEcoclass { value: EcoClass }
}

// program argument handeling
#[derive(Deserialize, Debug, Clone)]
struct Arguments {

    /// input .npz file
    input: PathBuf,

    /// input file for pixel per year values
    distances: PathBuf,

    /// input file for pixel per year values
    envelopes: PathBuf,

    /// output directory
    output_directory: PathBuf,

    /// dispersion directory
    dispersion_directory: PathBuf,

    /// temperature directory
    temperature_directory: PathBuf,

    /// precipitation directory
    precipitation_directory: PathBuf,

    /// number of iterations
    iters: usize,

    /// number of years
    years: usize,

    /// seed for RNG
    seed: u64,
}

impl Arguments {

    /// check the parsed arguments and setup any thing that
    /// the system will need (e.g. create output directories)
    pub fn setup(&self) {

        // check the input file
        if !self.input.exists() {
            eprintln!("input file does not exist!");
            ::std::process::exit(1);
        }

        // create the output directory
        if !self.output_directory.exists() {
            if let Err(_) = fs::create_dir_all(&self.output_directory) {
                eprintln!("Could Not Create Output Directory!");
                ::std::process::exit(2);
            }
        }

        // create the output directory
        if !self.dispersion_directory.exists() {
            if let Err(_) = fs::create_dir_all(&self.dispersion_directory) {
                eprintln!("Could Not Create Dispersion Directory!");
                ::std::process::exit(3);
            }
        }

        // create the output directory
        if !self.temperature_directory.exists() {
            if let Err(_) = fs::create_dir_all(&self.temperature_directory) {
                eprintln!("Could Not Create Temperature Directory!");
                ::std::process::exit(4);
            }
        }

        // create the output directory
        if !self.precipitation_directory.exists() {
            if let Err(_) = fs::create_dir_all(&self.precipitation_directory) {
                eprintln!("Could Not Create Precipiation Directory!");
                ::std::process::exit(5);
            }
        }

        // load csv of dispersion distances
        if !self.envelopes.exists() {
            eprintln!("envelope csv file does not exist!");
            ::std::process::exit(6);
        }

        // load csv of dispersion distances
        if !self.distances.exists() {
            eprintln!("pixel-per-year dispersion csv file does not exist!");
            ::std::process::exit(7);
        }
    }
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
fn update<T>(array: &mut Array2<T>, i: usize, j: usize) -> bool
    where T: Eq + Ord + Sync + Copy + Add + One + Zero {
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

fn load<P,T>(filename: P) -> Result<Array2<T>, Failure>
    where P: AsRef<Path>,
          T: ndarray_npy::ReadableElement {
    let f = fs::File::open(filename)?;
    let mut decoder = NpzReader::new(f)?;
    decoder.by_index(0).map_err(Into::into)
}

fn load_ecogrid<P: AsRef<Path>>(filename: P) -> Result<Array2<EcoClass>, Failure> {
    load::<P,EcoClass>(filename)
}


/// compute a distance grid for each value of the input grid. The
/// distance grid is the distance from each pixel of the grid to 
/// the pixels that represent a specific value on the grid. The
/// result is then divided by the pixel-per-year movement value
/// provided by an input csv file. The result is a data array for
/// each ecoclass that articulates how many years it will take to
/// move to a new location assuming "perfect" movement conditions.
fn compute_dispersion_rates(args: &Arguments) -> Result<(), Failure> {


    // load the csv table that contains pixel-per-year information
    // for each ecological class code.
    let dtable: HashMap<EcoClass, PixelPerYear> = load_dispersal_data(&args.distances)?;

    // load the ecological classes raster grid in numpy array format
    let ecogrid = load_ecogrid(&args.input)?;

    // get the dimensions of the ecological grid
    let (height, width): (usize, usize) = (ecogrid.shape()[0], ecogrid.shape()[1]);

    // harvest all of the unique values on the ecoclass grid
    let values = get_unique::<EcoClass>(&ecogrid);

    // ensure that each of the unique values has a
    // corresponding pixel-per-year distance in the
    // dispersion csv file.
    for v in values.iter() {
        if let None = dtable.get(&v) {
            return Err(Failure::MissingEcoclass { value: v.clone() });
        }
    }

    // compute the distance grid for each value of the
    // input grid. Then compute the pixel-per-year grid
    // for each value of the input grid. Write the result
    // to the output directory. Only write out the
    // pixel-per-year grids.
    values.par_iter()
          .for_each(|v|{

        // time the iteration
        let now = ::std::time::Instant::now();

        // construct a distance grid for this value
        // the grid is padded so that jacobi iteration
        // can be used to estimate distance using mahattan
        // distance measures. Distances are measured in
        // number of pixels.
        let eco = v.clone();
        let mut distance: Array2<u32> = Array2::from_elem((height+2, width+2), u32::max_value());
        for i in 0..height {
            for j in 0..width{
                distance[[i+1,j+1]] = if ecogrid[[i,j]] == eco { 0u32 } else { u32::max_value() - u32::one() };
            }
        }

        // determine the maximum number of jacbian iterations
        // that will be allowed based on the size of the input
        // data. This serves to limit any infinite looping.
        let max_iter_count = ((height as f32).powf(2f32) + (width as f32).powf(2f32)).sqrt() as usize;
        println!("predicted jacobian iteration count on ecoclass {0} is {1}", v, max_iter_count);

        // *** estimate distances using mahattan jocobian distance technique *** //

        // jacobi iteration
        let mut cntr = 0;
        loop {
            let mut do_loop = false;
            for i in 1..height {
                for j in 1..width {
                    do_loop = do_loop | update::<u32>(&mut distance, i, j);
                }
            }
            cntr += 1;
            if !do_loop { break; }
            if cntr > max_iter_count { eprintln!("did not converge!"); break; }
        }

        // helpful messages
        println!("convergence of jacobian iteration count on ecoclass {0} reach in {1} iters.", v, cntr);
        println!("distance estimation time for value {0} is {1}ms", v, now.elapsed().as_millis());
        
        // *** compute distance per pixel and for an output file *** //

        // reset the clock
        let now = ::std::time::Instant::now();

        // perform euclidean distance estimation as (PI/4 * Manhattan Distance)
        let mut estimates: Array2<Distance> = Array2::from_elem((height, width), 0.0);
        for i in 0..height {
            for j in 0..width {
                estimates[[i,j]] = (distance[[i,j]] as Distance) * (::std::f32::consts::PI/4.);
            }
        }

        // convert euclidean estimated distance to pixle-per-year results
        if let Some(pix_per_yr) = dtable.get(&v) {
            let mut dispersion: Array2<PixelPerYear> = Array2::from_elem((height, width), 0.0);
            for i in 0..height {
                for j in 0..width {
                    // convert distance units in pixels
                    // to the year since 0 that it will
                    // take to spread the ecoclass to a
                    // specific location on the grid.
                    dispersion[[i,j]] = estimates[[i,j]] / pix_per_yr;
                }
            }

            // produce an output array file for the estimated number
            // of years for this value to reach certain pixels.
            let outf = &args.dispersion_directory.join(format!("{}.npz", v));
            output_f32(&dispersion, &outf);
        } else {
            panic!("could not compute dispersion rate for {}", v);
        }

        // helpful messages
        println!("dispersion computation time for {0} is {1}", v, now.elapsed().as_millis());
    });

    Ok(())
}


fn main() {

    // load options
    let mut settings = config::Config::default();
    settings.merge(config::File::with_name("config")).expect("failed to load config!");
    let opt = settings.deserialize::<Arguments>().expect("failed to deserialize arguments!");
    println!("{:#?}", opt);

    // check and setup arguments
    opt.setup();

    // compute a dispersion grid for each array
    println!("compute dispersion grids");
    if let Err(e) = compute_dispersion_rates(&opt) {
        eprintln!("error: {:?}", e);
    }

    // compute the switches
    println!("compute switches");
    if let Err(e) = compute_switches(&opt) {
        eprintln!("error: {:?}", e);
    }
}

fn compute_switches(args: &Arguments) -> Result<(), Failure> {

    println!("load data:");
    // load the climate envelope csv files
    let envelopes = load_envelope_data(&args.envelopes)?;
    println!("[+] envelopes loaded");

    // load the original ecoclass grid
    let ecogrid = load_ecogrid(&args.input)?;
    println!("[+] ecogrid loaded");

    // load the temperature data
    let temperatures = load_temperature_data(&args.temperature_directory)?;
    println!("[+] temperature loaded");

    // load the precipitation data
    let precipitation = load_precipitation_data(&args.precipitation_directory)?;
    println!("[+] precipitation loaded");

    // load the dispersion data
    let dispersion = load_dispersion_data(&args.dispersion_directory)?;
    println!("[+] dispersion loaded");

    // get the dimensions of the ecological grid
    let (height, width): (usize, usize) = (ecogrid.shape()[0], ecogrid.shape()[1]);

    // perform random walk
    println!("compute random walk!");
    (0..args.iters)
        .collect::<Vec<usize>>()
        .par_iter()
        .for_each(|iteration|{
            // time the iteration
            let now = ::std::time::Instant::now();

            // form output grid for iteration
            let mut output = ecogrid.clone();

            // set RNG
            let mut rng = rand_isaac::isaac::IsaacRng::seed_from_u64( args.seed + (*iteration as u64));

            // iteration output directory
            let outdir = &args.output_directory.join(format!("iteration-{}", iteration));
            if !outdir.exists() { fs::create_dir_all(&outdir); }

            for year in 0..args.years {

                // get the yearly climate
                let temp = &temperatures[&(year as Year)];
                let precip = &precipitation[&(year as Year)];

                // perform random walk cell by cell
                for w in 0..width {
                    for h in 0..height {
                        // get the current condition and test if it is outside its envelope
                        if let Some((tmx,tmn,pmx,pmn)) = envelopes.get(&output[[w,h]]) {
                            // get current climate
                            let t = temp[[w,h]];
                            let p = precip[[w,h]];
                            // is current climate outside the envelope?
                            if t < *tmn || t > *tmx || p > *pmx || p < *pmn {
                                let mut switches = vec![];
                                // determine potential switches
                                for (eco, array) in dispersion.iter() {
                                    // filter by dispersion distances
                                    if array[[w,h]] <= (year as f32) {
                                        // filter potentials by envelopes
                                        if let Some((tmx2, tmn2, pmx2, pmn2)) = envelopes.get(&eco) {
                                            if t >= *tmn2 && t <= *tmx2 && t <= *pmx2 && t >= *pmn2 {
                                                switches.push(eco.clone());
                                            }
                                        }
                                    }
                                }
                                // randomly select a new value
                                if let Some(selected) = switches.choose(&mut rng) {
                                    output[[w,h]] = selected.clone();
                                }
                            }
                        }
                    }
                }

                // produce the output path
                let outpath = &outdir.join(format!("year-{}.npz", year));
                // produce the output ecological region
                output_eco(&output, &outpath);
                
            }
            
        // helpful messages
        println!("iteration {0} completed in {1}ms.", iteration, now.elapsed().as_millis());
    });

    // default return
    Ok(())
}

/// load the temperature data from the directory
fn load_temperature_data<P: AsRef<Path>>(directory: P) -> Result<HashMap<Year, Array2<Temp>>, Failure> {
    load_directory_as_ints::<P,Temp,Year>(directory)
}

/// load the precipitation data from the directory
fn load_precipitation_data<P: AsRef<Path>>(directory: P) -> Result<HashMap<Year, Array2<Precip>>, Failure> {
    load_directory_as_ints::<P,Precip,Year>(directory)
}

/// load the dispersion data from the directory
fn load_dispersion_data<P: AsRef<Path>>(directory: P) -> Result<HashMap<EcoClass, Array2<PixelPerYear>>, Failure> {
    load_directory_as_ints::<P,PixelPerYear,EcoClass>(directory)
}


/// load directory where file-stems are years
fn load_directory_as_ints<P,T,K>(directory: P) -> Result<HashMap<K, Array2<T>>, Failure> 
    where P: AsRef<Path>,
          T: ndarray_npy::ReadableElement,
          K: Hash + Copy + FromStr + num::Num + Eq,
          Failure: From<<K as FromStr>::Err> {

    // construct output hash map
    let mut hm: HashMap<K, Array2<T>> = HashMap::new();

    // iterate over file in the directory
    let indir = PathBuf::from(directory.as_ref());
    let mut iter = indir.read_dir()?;
    while let Some(Ok(entry)) = iter.next() {

        // get filepath of entry
        let f = entry.path();

        // parse int from file name
        let yr: K = determine_int(&f)?;

        // load each file
        let array = load::<&PathBuf,T>(&f)?;

        // insert loaded array into hashmap indexed by year
        hm.insert(yr, array);
    }

    // return result
    Ok(hm)
}

/// determine the int of a file from a file-stem
fn determine_int<P: AsRef<Path>, T: FromStr>(path: P) -> Result<T, Failure> where Failure: From<<T as FromStr>::Err> {
    // get path buf
    let pb = PathBuf::from(path.as_ref());
    // get basename
    Ok(T::from_str(pb.file_stem()?.to_str()?)?)
}

fn output_f32<P: AsRef<Path>>(array: &Array2<f32>, path: P) -> Result<(), Failure> {
    output::<P, f32>(path, &array)
}

fn output_eco<P: AsRef<Path>>(array: &Array2<EcoClass>, path: P) -> Result<(), Failure> {
    output::<P, EcoClass>(path, &array)
}

fn output<P,T>(path: P, array: &Array2<T>) -> Result<(), Failure> 
    where P: AsRef<Path>,
          T: ndarray_npy::WritableElement {
    // open a file to write to
    let writer = fs::File::create(path)?;
    // construct the output data
    let mut npz = NpzWriter::new_compressed(writer);
    // write the data
    npz.add_array("a", &array)?;
    // default return
    Ok(())
}

/// load the csv data for the spreading functions
fn load_dispersal_data<P: AsRef<Path>>(path: P) -> Result<HashMap<EcoClass, f32>, Failure> {
    #[derive(Deserialize)]
    struct Record {
        // code for the ecological region that is stored on the raster
        pub code: EcoClass,
        // distance the ecological region can move in pixels per year
        pub dist: f32,
    }
    // construct output
    let mut table: HashMap<EcoClass, f32> = HashMap::new();
    // open the csv file
    if let Ok(ref f) = fs::File::open(path.as_ref()) {
        let mut rdr = csv::Reader::from_reader(f);
        for result in rdr.deserialize::<Record>() {
            if let Ok(record) = result {
                table.insert(record.code, record.dist);
            }
        }
    }
    // return the table
    Ok(table)
}

/// load the csv data for the envelope data
fn load_envelope_data<P>(path: P) -> Result<HashMap<EcoClass, (Temp,Temp,Precip,Precip)>, Failure> 
    where P: AsRef<Path> {

    #[derive(Deserialize)]
    struct Envelope {
        // ecocode
        pub code: EcoClass,
        // max temp
        pub temp_mx: Temp,
        // min temp
        pub temp_mn: Temp,
        // max precip
        pub precip_mx: Precip,
        // min precip
        pub precip_mn: Precip,
    }
    // construct output
    let mut table: HashMap<EcoClass, (Temp,Temp,Precip,Precip)> = HashMap::new();
    // open the csv file
    if let Ok(ref f) = fs::File::open(path.as_ref()) {
        let mut rdr = csv::Reader::from_reader(f);
        for result in rdr.deserialize::<Envelope>() {
            if let Ok(record) = result {
                table.insert(record.code, (record.temp_mx, record.temp_mn, record.precip_mx, record.precip_mn));
            }
        }
    }
    // return the table
    Ok(table)
}