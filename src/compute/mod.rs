use super::*;

/// # DistanceTable
/// Computes a lookup table of distances over a given width and height
pub struct DistanceTable(usize, usize, BTreeMap<(usize,usize), f32>);

impl DistanceTable {

    /// construct a new lookup table
    pub fn new(width: usize, height: usize) -> DistanceTable {

        // get max quadrant size
        let size: usize = max(width, height);

        // allocate space for BTreeMap
        let mut btmap = BTreeMap::new();

        // generate all point pairs for a single quadrant
        let points: Vec<(usize, usize)> = (0..size)
                 .into_par_iter()
                 .flat_map(|i| 
                    (i..size).map(|j| (i,j))
                             .collect::<Vec<(usize,usize)>>()
                 )
                 .collect();

        // compute all distances to each of the point pairs
        // and load the btree map
        for d in points.par_iter()
                       .map(Self::euclid)
                       .collect::<Vec<(usize,usize,f32)>>()
                       .drain(..) {
            btmap.insert((d.0,d.1), d.2);
        }

        // return the QuadrantDistanceTable
        DistanceTable(width, height, btmap)
    }

    /// lookup the distance value between two points
    pub fn distance(&self, mut a: (usize, usize), mut b: (usize, usize)) -> f32 {
        self.lookup((a.0 as i32 - b.0 as i32).abs() as usize, (a.1 as i32 - b.1 as i32).abs() as usize)
    }

    /// lookup the distance value
    pub fn lookup(&self, mut i: usize, mut j: usize) -> f32 {
        // lookup the value
        if i <= j {
            match self.2.get(&(i,j)) {
                Some(k) => *k,
                None => -1.
            }
        } else {
            match self.2.get(&(j,i)) {
                Some(k) => *k,
                None => -1.
            }
        }
    }

    // helper method for distance from origin
    fn euclid(p: &(usize, usize)) -> (usize, usize, f32) {
        (
            p.0,
            p.1,
            ((p.0 as f32).powf(2.) +
            (p.1 as f32).powf(2.))
            .sqrt()
        )
    }
}

/// get the edge points of a value on a given array
pub fn get_edge_points<T: Eq + Sync>(array: Array2<T>, value: T) -> Vec<(usize,usize)> {

    // get array dimensions
    let height = array.shape()[0];
    let width = array.shape()[1];

    // iterate over the array looking for edge values
    (0..height-1).into_par_iter()
                 .flat_map(|i| unsafe {

                    // allocate output
                    let mut output = vec![];

                    // allocate window
                    let mut window: [bool; 3] = [false, false, false];

                    for j in 0..width {
                        // check for value at location
                        if *array.uget([i,j]) != value { continue; }
                        // load left
                        if j != 0 {
                            window[0] = *array.uget([i, j-1]) != value;
                        } else { window[0] = false; }
                        // load bottom
                        window[1] = *array.uget([i+1, j]) != value;
                        // load top
                        if i != 0 {
                            window[2] = *array.uget([i-1, j]) != value;
                        } else { window[2] = false; }
                        // check for edge value
                        if window[0] || window[1] || window[2] {
                            output.push((i,j));
                        } 
                    }
                    output
    })
    .collect::<Vec<(usize,usize)>>()
}

/// get the unique values on the array
pub fn get_unique<T: Eq + Hash + Clone>(array: &Array2<T>) -> Vec<T> {
    let mut set: HashSet<T> = HashSet::new();
    for v in array.into_iter() {
        set.insert(v.clone());
    }
    set.into_iter().collect()
}