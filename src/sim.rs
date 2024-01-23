use std::time::Instant;

use eigenvalues::{
    davidson::Davidson, lanczos::HermitianLanczos, matrix_operations::MatrixOperations,
    DavidsonCorrection, SpectrumTarget,
};
use glam::Vec2;

use linfa_linalg::lobpcg::LobpcgResult;
use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

// TODO: Set these parameters ...
const NUCLEAR_MASS: f32 = 1.0;
const ELECTRON_MASS: f32 = 1.0;
const HBAR: f32 = 1.0;

pub type Grid2D<T> = Array2<T>;

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
    pub tolerance: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Nucleus {
    // pub mass: f32,
    /// Velocity
    pub vel: Vec2,
    /// Position
    pub pos: Vec2,
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
    pub eigenstates: Vec<Grid2D<f32>>,
    /// Energy levels (E_n)
    pub energies: Vec<f32>,
    /// Map of electric potential due to nuclei
    pub potential: Grid2D<f32>,
    // /// Amount of energy in the classical system at present time
    // pub classical_energy: f32,
}

pub struct Sim {
    pub cfg: SimConfig,
    pub state: SimState,
    cache: Option<Cache>,
    artefacts: Option<SimArtefacts>,
}

impl Sim {
    pub fn new(cfg: SimConfig, init_state: SimState) -> Self {
        let mut inst = Self {
            state: init_state,
            cfg,
            cache: None,
            artefacts: None,
        };
        inst.recalculate();
        inst
    }

    pub fn recalculate(&mut self) {
        let (artefacts, cache) = calculate_artefacts(&self.cfg, &self.state, self.cache.take());
        self.cache = Some(cache);
        self.artefacts = Some(artefacts);
    }

    pub fn state(&self) -> &SimState {
        &self.state
    }

    pub fn artefacts(&self) -> Option<&SimArtefacts> {
        self.artefacts.as_ref()
    }

    pub fn cfg(&self) -> &SimConfig {
        &self.cfg
    }

    pub fn step(&mut self) {}
}

fn calculate_artefacts(
    cfg: &SimConfig,
    state: &SimState,
    cache: Option<Cache>,
) -> (SimArtefacts, Cache) {
    let potential = calculate_potential(cfg, state);
    let (energies, eigenstates, cache) = solve_schrödinger(cfg, &potential, cache);

    (
        SimArtefacts {
            eigenstates,
            energies,
            potential,
        },
        cache,
    )
}

/// Calculate the electric potential in the position basis
fn calculate_potential(cfg: &SimConfig, state: &SimState) -> Grid2D<f32> {
    let grid_positions = grid_positions(cfg);

    let v = grid_positions
        .iter()
        .map(|grid_pos| {
            state
                .nuclei
                .iter()
                .map(|nucleus| {
                    let r = (nucleus.pos - *grid_pos).length();
                    softened_potential(r, cfg)
                })
                .sum()
        })
        .collect();

    Grid2D::from_shape_vec((cfg.grid_width, cfg.grid_width), v).unwrap()
}

fn softened_potential(r: f32, cfg: &SimConfig) -> f32 {
    cfg.v0 / (r * cfg.v_scale + cfg.v_soft)
}

/// World-space positions at grid points
fn grid_positions(cfg: &SimConfig) -> Grid2D<Vec2> {
    let mut output = Grid2D::from_shape_vec(
        (cfg.grid_width, cfg.grid_width),
        vec![Vec2::ZERO; cfg.grid_width.pow(2)],
    )
    .unwrap();

    for y in 0..cfg.grid_width {
        for x in 0..cfg.grid_width {
            output[(x, y)] = Vec2::new(x as f32, y as f32) * cfg.dx;
        }
    }

    output
}

/// Returns false if out of bounds with the given width
fn bounds_check(pt_x: i32, pt_y: i32, width: i32) -> Option<(usize, usize)> {
    (pt_x >= 0 && pt_y >= 0 && pt_x < width && pt_y < width).then(|| (pt_x as usize, pt_y as usize))
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
struct HamiltonianObject {
    potential: Grid2D<f32>,
    cfg: SimConfig,
}

impl HamiltonianObject {
    pub fn from_potential(potential: &Grid2D<f32>, cfg: &SimConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            // Diagonal includes both the potential AND the stencil centers
            potential: potential.clone(),
        }
    }
}

impl HamiltonianObject {
    fn ncols(&self) -> usize {
        self.potential.nrows() * self.potential.ncols()
    }

    fn nrows(&self) -> usize {
        self.ncols()
    }

    fn matrix_vector_prod(&self, psi: Grid2D<f32>) -> Grid2D<f32> {
        let mut output = Grid2D::zeros((self.cfg.grid_width, self.cfg.grid_width));

        for y in 0..psi.nrows() {
            for x in 0..psi.ncols() {
                let center_grid_coord = (x, y);

                let mut sum = 0.0;

                let pot = self.potential[center_grid_coord];

                for (off, coeff) in [
                    ((-1, 0), 1.0),
                    ((1, 0), 1.0),
                    ((0, 1), 1.0),
                    ((0, -1), 1.0),
                    ((0, 0), pot - 4.0),
                    //(Vec2::new(0, 0), 1.0),
                ] {
                    if let Some(grid_coord) =
                        bounds_check(x as i32 + off.0, y as i32 + off.1, psi.ncols() as i32)
                    {
                        sum += coeff * psi[grid_coord];
                    }
                }

                let kinetic = sum; // * (-HBAR / ELECTRON_MASS / 2.0 / self.cfg.dx.powi(2));

                //output[center_grid_coord] = kinetic; //kinetic + self.potential[center_grid_coord];
                output[center_grid_coord] = kinetic; //kinetic + self.potential[center_grid_coord];
            }
        }

        output
    }

    // NOTE: This operation is not in the hot path so it is NOT optimized!
    fn matrix_matrix_prod(&self, mut mtx: Array2<f32>) -> Array2<f32> {
        for mut column in mtx.columns_mut() {
            let res = self.matrix_vector_prod(vector_to_state(&column.to_owned(), &self.cfg));
            column.assign(&res);
        }
        mtx
    }
}

/*
fn hamiltonian_flat(
cfg: &SimConfig,
potential: &StateVector,
flat_input_vect: &[Complex64],
flat_output_vect: &mut [Complex64],
) {
let psi = SpatialVector2D::from_array(potential.ncols(), flat_input_vect.to_vec());
let output = hamiltonian(cfg, &psi, potential);
flat_output_vect.copy_from_slice(output.data());
}
*/

/// Eigenvectors
pub type Cache = Array2<f32>;

/// Solves the Schrödinger equation for the first N energy eigenstates
///
/// Generates the second-derivative finite-difference stencil in the position basis. This is then
/// combined with the potential to form the Hamiltonian.
fn solve_schrödinger(
    cfg: &SimConfig,
    potential: &Grid2D<f32>,
    cache: Option<Cache>,
) -> (Vec<f32>, Vec<Grid2D<f32>>, Cache) {
    let ham = HamiltonianObject::from_potential(potential, cfg);

    let preconditioner: Array2<f32> = match cache {
        None => Array2::random((ham.ncols(), cfg.n_states), Uniform::new(-1.0, 1.0)),
        Some(cache) => cache,
    };

    let result = linfa_linalg::lobpcg::lobpcg::<f32, _, _>(
        |vects| ham.matrix_matrix_prod(vects.to_owned()),
        preconditioner,
        |_| (),
        None,
        cfg.tolerance,
        cfg.num_solver_iters,
        linfa_linalg::lobpcg::Order::Smallest,
    );

    let eigvects: Vec<Grid2D<f32>>;
    let eigvals: Vec<f32>;
    let cache;
    match result {
        LobpcgResult::Ok(eig) | LobpcgResult::Err((_, Some(eig))) => {
            eigvals = eig.eigvals.as_slice().unwrap().to_vec();

            eigvects = eig
                .eigvecs
                .columns()
                .into_iter()
                .map(|col| vector_to_state(&col.to_owned(), cfg))
                .collect();

            cache = eig.eigvecs;
        }
        LobpcgResult::Err((e, None)) => panic!("{}", e),
    }

    // Sort by energy
    let mut indices: Vec<_> = eigvals.iter().copied().zip(eigvects).collect();
    indices.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (sorted_eigvals, sorted_eigvecs) = indices.into_iter().unzip();

    (sorted_eigvals, sorted_eigvecs, cache)
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

#[cfg(test)]
mod tests {
    use crate::initial_cfg;

    use super::*;

    #[test]
    fn roundtrip_nalgebra_to_array() {
        let w = 20;
        let nalg = DVector::from_iterator(w, (0..w).map(|i| i as f32));
        let arr = ((&nalg).into(), &initial_cfg());
        assert_eq!(nalg, state_to_vector(&arr));
    }
}

impl Default for Nucleus {
    fn default() -> Self {
        Self {
            vel: Vec2::ZERO,
            pos: Vec2::ZERO,
        }
    }
}

/*
fn state_to_vector(state: &Grid2D<f32>) -> Array1<f32> {
    state.clone().into_shape(state.nrows() * state.ncols()).unwrap()
}
*/

fn vector_to_state(state: &Array1<f32>, cfg: &SimConfig) -> Grid2D<f32> {
    Array2::from_shape_vec((cfg.grid_width, cfg.grid_width), state.iter().copied().collect()).unwrap()
}
