use std::time::Instant;

use eigenvalues::{
    davidson::Davidson, lanczos::HermitianLanczos, matrix_operations::MatrixOperations,
    DavidsonCorrection, SpectrumTarget,
};
use nalgebra::{
    ComplexField, DMatrix, DMatrixSlice, DVector, DVectorSlice, MatrixN, Point2, SymmetricEigen,
    Vector2,
};

use linfa_linalg::lobpcg::LobpcgResult;

use crate::{
    array2d::Array2D,
    lobster::{lobster, LobsterOutput},
};

// TODO: Set these parameters ...
const NUCLEAR_MASS: f32 = 1.0;
const ELECTRON_MASS: f32 = 1.0;
const HBAR: f32 = 1.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EigenAlgorithm {
    Nalgebra,
    Lanczos,
    LobPcg,

    Lobster,
}

#[derive(Clone)]
pub struct SimConfig {
    // Options which should NOT change during runtime
    /// Spacing between adjacent points on the grid
    pub dx: f32,
    /// Grid width
    pub grid_width: usize,

    // Options which should may be able to change during runtime...
    /// Depth of potential wells due to nuclei
    pub v0: f32,
    /// Potential function softening factor
    pub v_soft: f32,
    /// Potential function scale factor
    pub v_scale: f32,

    /// Number of eigenstates needed
    pub n_states: usize,
    pub num_solver_iters: usize,
    // /// Mass of each nucleus
    pub eig_algo: EigenAlgorithm,
    pub tolerance: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Nucleus {
    // pub mass: f32,
    /// Velocity
    pub vel: Vector2<f32>,
    /// Position
    pub pos: Point2<f32>,
}

#[derive(Clone, Debug)]
pub struct SimState {
    /// Index of the quantum energy level (lambda)
    pub energy_level: usize,
    /// Wavefunction coefficients (c_j)
    pub coeffs: Vec<f32>,
    /// Atomic nuclei (R, P)
    pub nuclei: Vec<Nucleus>,
}

/// Data uniquely generated from a SimState
#[derive(Clone)]
pub struct SimArtefacts {
    /// Energy eigenstates (psi_n)
    pub eigenstates: Vec<Array2D<f32>>,
    /// Energy levels (E_n)
    pub energies: Vec<f32>,
    /// Map of electric potential due to nuclei
    pub potential: Array2D<f32>,
    // /// Amount of energy in the classical system at present time
    // pub classical_energy: f32,
}

pub struct Sim {
    cfg: SimConfig,
    state: SimState,
    artefacts: SimArtefacts,
}

impl Sim {
    pub fn new(cfg: SimConfig, state: SimState, cache: Option<SimArtefacts>) -> Self {
        let artefacts = calculate_artefacts(&cfg, &state, cache);

        Self {
            state,
            cfg,
            artefacts,
        }
    }

    pub fn state(&self) -> &SimState {
        &self.state
    }

    pub fn artefacts(&self) -> &SimArtefacts {
        &self.artefacts
    }

    pub fn cfg(&self) -> &SimConfig {
        &self.cfg
    }
}

fn calculate_artefacts(
    cfg: &SimConfig,
    state: &SimState,
    cache: Option<SimArtefacts>,
) -> SimArtefacts {
    let potential = calculate_potential(cfg, state);
    let (energies, eigenstates) = solve_schrödinger(cfg, &potential, cache);

    SimArtefacts {
        eigenstates,
        energies,
        potential,
    }
}

/// Calculate the electric potential in the position basis
fn calculate_potential(cfg: &SimConfig, state: &SimState) -> Array2D<f32> {
    let grid_positions = grid_positions(cfg);

    let v = grid_positions
        .data()
        .iter()
        .map(|grid_pos| {
            state
                .nuclei
                .iter()
                .map(|nucleus| {
                    let r = (nucleus.pos - grid_pos).magnitude();
                    softened_potential(r, cfg)
                })
                .sum()
        })
        .collect();

    Array2D::from_array(cfg.grid_width, v)
}

fn softened_potential(r: f32, cfg: &SimConfig) -> f32 {
    cfg.v0 / (r * cfg.v_scale + cfg.v_soft)
}

/// World-space positions at grid points
fn grid_positions(cfg: &SimConfig) -> Array2D<Point2<f32>> {
    let mut output = Array2D::from_array(
        cfg.grid_width,
        vec![Point2::origin(); cfg.grid_width.pow(2)],
    );

    for y in 0..cfg.grid_width {
        for x in 0..cfg.grid_width {
            output[(x, y)] = Point2::new(x as f32, y as f32) * cfg.dx;
        }
    }

    output
}

/// Returns false if out of bounds with the given width
fn bounds_check(pt: Point2<i32>, width: i32) -> Option<(usize, usize)> {
    (pt.x >= 0 && pt.y >= 0 && pt.x < width && pt.y < width).then(|| (pt.x as usize, pt.y as usize))
}

/*
///
///                 H = (-ħ/2m)∇² + V
///
/// In the 2D finite difference the stencil looks like:
///                   | 0   k   0 |
///                   | k  V-4k k | where k = -ħ/2m
///                   | 0   k   0 |
*/
#[derive(Clone)]
pub struct HamiltonianObject {
    potential: Array2D<f32>,
    cfg: SimConfig,
}

impl HamiltonianObject {
    pub fn from_potential(potential: &Array2D<f32>, cfg: &SimConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            // Diagonal includes both the potential AND the stencil centers
            potential: potential.clone(),
        }
    }
}

fn nalgebra_to_array2d(vs: DVectorSlice<f32>, cfg: &SimConfig) -> Array2D<f32> {
    Array2D::from_array(cfg.grid_width, vs.as_slice().to_vec())
}

fn array2d_to_nalgebra(arr: &Array2D<f32>) -> DVector<f32> {
    arr.data().to_vec().into()
}

impl HamiltonianObject {
    fn ncols(&self) -> usize {
        self.potential.data().len()
    }

    fn nrows(&self) -> usize {
        self.potential.data().len()
    }

    fn diagonal(&self) -> DVector<f32> {
        unimplemented!()
    }

    fn set_diagonal(&mut self, _diag: &DVector<f32>) {
        unimplemented!()
    }

    fn matrix_vector_prod(&self, vs: DVectorSlice<f32>) -> DVector<f32> {
        let psi = nalgebra_to_array2d(vs, &self.cfg);

        let mut output = Array2D::new(self.cfg.grid_width, self.cfg.grid_width);

        for x in 0..psi.width() {
            for y in 0..psi.height() {
                let center_world_coord = Point2::new(x as i32, y as i32);
                let center_grid_coord = (x, y);

                let mut sum = 0.0;

                let pot = self.potential[center_grid_coord];

                for (off, coeff) in [
                    (Vector2::new(-1, 0), 1.0),
                    (Vector2::new(1, 0), 1.0),
                    (Vector2::new(0, 1), 1.0),
                    (Vector2::new(0, -1), 1.0),
                    (Vector2::new(0, 0), pot - 4.0),
                    //(Vector2::new(0, 0), 1.0),
                ] {
                    if let Some(grid_coord) =
                        bounds_check(center_world_coord + off, psi.width() as i32)
                    {
                        sum += coeff * psi[grid_coord];
                    }
                }

                let kinetic = sum; // * (-HBAR / ELECTRON_MASS / 2.0 / self.cfg.dx.powi(2));

                //output[center_grid_coord] = kinetic; //kinetic + self.potential[center_grid_coord];
                output[center_grid_coord] = kinetic; //kinetic + self.potential[center_grid_coord];
            }
        }

        array2d_to_nalgebra(&output)
    }

    // NOTE: This operation is not in the hot path so it is NOT optimized!
    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<f32>) -> DMatrix<f32> {
        let mut out_cols = vec![];
        for in_column in mtx.column_iter() {
            out_cols.push(self.matrix_vector_prod(in_column));
        }
        DMatrix::from_columns(&out_cols)
    }
}

/*
fn hamiltonian_flat(
cfg: &SimConfig,
potential: &Array2D<f32>,
flat_input_vect: &[Complex64],
flat_output_vect: &mut [Complex64],
) {
let psi = Array2D::from_array(potential.width(), flat_input_vect.to_vec());
let output = hamiltonian(cfg, &psi, potential);
flat_output_vect.copy_from_slice(output.data());
}
*/

/// Solves the Schrödinger equation for the first N energy eigenstates
///
/// Generates the second-derivative finite-difference stencil in the position basis. This is then
/// combined with the potential to form the Hamiltonian.
fn solve_schrödinger(
    cfg: &SimConfig,
    potential: &Array2D<f32>,
    cache: Option<SimArtefacts>,
) -> (Vec<f32>, Vec<Array2D<f32>>) {
    assert_eq!(cfg.grid_width, potential.width());

    let cache = cache.filter(|cache| cache.energies.len() == cfg.n_states);
    let cache = cache.filter(|cache| cache.potential.width() == potential.width());

    // Build the Hamiltonian
    let ham = HamiltonianObject::from_potential(potential, cfg);

    // Calculate energy eigenstates
    //let start = Instant::now();

    let eigvects: Vec<Array2D<f32>>;
    let eigvals: Vec<f32>;

    match cfg.eig_algo {
        EigenAlgorithm::Nalgebra => {
            let ident = DMatrix::identity(potential.data().len(), potential.data().len());
            let ham_matrix = ham.matrix_matrix_prod((&ident).into());

            let eig = SymmetricEigen::new(ham_matrix.clone());

            eigvects = eig
                .eigenvectors
                .column_iter()
                .map(|col| nalgebra_to_array2d(col, cfg))
                .collect();

            eigvals = eig.eigenvalues.as_slice().to_vec();
        }
        EigenAlgorithm::Lanczos => {
            /*
            let eig = HermitianLanczos::new(ham, cfg.n_states, SpectrumTarget::Lowest).unwrap();

            eigvects = eig
                .eigenvectors
                .column_iter()
                .map(|col| nalgebra_to_array2d(col, cfg))
                .collect();

            eigvals = eig.eigenvalues.as_slice().to_vec();
            */
            unimplemented!()
        }
        EigenAlgorithm::LobPcg => {
            use ndarray_rand::{rand_distr::Uniform, RandomExt};
            let x = match cache {
                None => {
                    ndarray::Array::random((ham.ncols(), cfg.n_states), Uniform::new(-1.0, 1.0))
                }
                Some(cache) => {
                    let vects: Vec<DVector<f32>> = cache
                        .eigenstates
                        .into_iter()
                        .map(|state| array2d_to_nalgebra(&state))
                        .collect();
                    nalgebra_to_ndarray(DMatrix::from_columns(vects.as_slice()))
                }
            };

            let result = linfa_linalg::lobpcg::lobpcg::<f32, _, _>(
                |vects| {
                    let vects: DMatrix<f32> = ndarray_to_nalgebra(vects.to_owned());
                    let prod = ham.matrix_matrix_prod((&vects).into());
                    nalgebra_to_ndarray(prod)
                },
                x,
                |_| (),
                None,
                cfg.tolerance,
                cfg.num_solver_iters,
                linfa_linalg::lobpcg::Order::Smallest,
            );

            match result {
                LobpcgResult::Ok(eig) | LobpcgResult::Err((_, Some(eig))) => {
                    eigvals = eig.eigvals.as_slice().unwrap().to_vec();

                    eigvects = eig
                        .eigvecs
                        .columns()
                        .into_iter()
                        .map(|col| {
                            nalgebra_to_array2d(
                                (&ndarray_to_nalgebra_vect(col.to_owned())).into(),
                                cfg,
                            )
                        })
                        .collect();
                }
                LobpcgResult::Err((e, None)) => panic!("{}", e),
            }
        }
        EigenAlgorithm::Lobster => {
            let lob = lobster(ham);
            eigvals = lob.eigenvalues;
            eigvects = lob.eigenvectors;
        }
    };
    //let time = start.elapsed().as_secs_f32();
    //dbg!(time);

    // Sort by energy
    let mut indices: Vec<_> = eigvals.iter().copied().zip(eigvects).collect();
    indices.sort_by(|a, b| a.0.total_cmp(&b.0));

    indices.into_iter().unzip()
    //let (energies, eigvects) = indices.iter().map(|((idx, val), eigve| *val).collect();

    /*
    // DEBUGGGING
    let sel_idx = 0;
    let sel_energy = eig.eigenvalues[sel_idx];
    let sel_energy_eigenstate = eig.eigenvectors.column(sel_idx);
    let hpsi = &ham_matrix * sel_energy_eigenstate;

    let expect = sel_energy * sel_energy_eigenstate;

    let percent_err = (hpsi - &expect).abs().component_div(&expect.abs());
    let avg_err = percent_err.sum() / percent_err.len() as f32;

    //dbg!(percent_err);
    dbg!(avg_err);
    dbg!(&eig.eigenvalues[sel_idx]);
    */
    //(eigvals, eigvects)
}

/*
fn calculate_classical_energy(cfg: &SimConfig, state: &SimState) -> f32 {
let kinetic_energy: f32 = state
.nuclei
.iter()
.map(|nucleus| NUCLEAR_MASS * nucleus.vel.magnitude_squared() / 2.)
.sum();

todo!()
}
*/

fn nalgebra_to_ndarray(vect: nalgebra::DMatrix<f32>) -> ndarray::Array2<f32> {
    ndarray::Array2::from_shape_vec(vect.shape(), vect.transpose().as_slice().to_vec()).unwrap()
}

fn ndarray_to_nalgebra(vect: ndarray::Array2<f32>) -> nalgebra::DMatrix<f32> {
    DMatrix::from_iterator(vect.nrows(), vect.ncols(), vect.t().iter().copied())
}

fn ndarray_to_nalgebra_vect(vect: ndarray::Array1<f32>) -> nalgebra::DVector<f32> {
    DVector::from_vec(vect.as_slice().unwrap().to_vec())
}

impl Sim {
    pub fn step(&mut self) {}
}

#[cfg(test)]
mod tests {
    use crate::initial_cfg;

    use super::*;

    #[test]
    fn roundtrip_nalgebra_to_array() {
        let w = 20;
        let nalg = DVector::from_iterator(w, (0..w).map(|i| i as f32));
        let arr = nalgebra_to_array2d((&nalg).into(), &initial_cfg());
        assert_eq!(nalg, array2d_to_nalgebra(&arr));
    }
}

impl Default for Nucleus {
    fn default() -> Self {
        Self {
            vel: Vector2::zeros(),
            pos: Point2::origin(),
        }
    }
}

impl HamiltonianObject {
    /// Create an operator which is half the resolution of this one.
    pub fn half(&self) -> Self {
        let mut cfg = self.cfg.clone();
        cfg.grid_width /= 2;

        Self {
            cfg,
            potential: linear_downscale_by_two(&self.potential),
        }
    }
}

pub fn linear_downscale_by_two(inp: &Array2D<f32>) -> Array2D<f32> {
    let mut outp = Array2D::new(inp.width() / 2, inp.height() / 2);

    for y in 0..inp.height() {
        for x in 0..inp.width() {
            let x_base = x * 2;
            let y_base = y * 2;
            outp[(x, y)] = (inp[(x_base, y_base)]
                + inp[(x_base + 1, y_base)]
                + inp[(x_base, y_base + 1)]
                + inp[(x_base + 1, y_base + 1)])
                / 4.0;
        }
    }

    outp
}

pub fn nearest_upscale_by_two(inp: &Array2D<f32>) -> Array2D<f32> {
    let mut outp = Array2D::new(inp.width() * 2, inp.height() * 2);

    for y in 0..outp.height() {
        for x in 0..outp.width() {
            outp[(x, y)] = inp[(x / 2, y / 2)];
        }
    }

    outp
}
