use eigenvalues::matrix_operations::MatrixOperations;
use glam::Vec2;

use linfa_linalg::lobpcg::LobpcgResult;
use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

// TODO: Set these parameters ...
const NUCLEAR_MASS: f32 = 1836.2; // μ = Mp/Me
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

    pub potental_mode: PotentialMode,
    pub eigval_search: linfa_linalg::lobpcg::Order,
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
pub struct SimElectronicState {
    /// Energy eigenstates (psi_n)
    pub eigenstates: Vec<Grid2D<f32>>,
    /// Energy levels (E_n)
    pub energies: Vec<f32>,
    /// Hamiltonian object
    pub ham: HamiltonianObject,
    // /// Amount of energy in the classical system at present time
    // pub classical_energy: f32,
}

pub struct Sim {
    pub cfg: SimConfig,
    pub state: SimState,
    cache: Option<Cache>,
    pub elec_state: Option<SimElectronicState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PotentialMode {
    Delta,
    Kqr,
}

impl Sim {
    pub fn new(cfg: SimConfig, init_state: SimState) -> Self {
        let mut inst = Self {
            state: init_state,
            cfg,
            cache: None,
            elec_state: None,
        };
        inst.recalculate();
        inst
    }

    pub fn clear_cache(&mut self) {
        self.cache = None;
    }

    pub fn recalculate(&mut self) {
        let (elec_state, cache) = calculate_electric_state(&self.cfg, &self.state, self.cache.take());
        self.cache = Some(cache);
        self.elec_state = Some(elec_state);
    }

    pub fn state(&self) -> &SimState {
        &self.state
    }

    pub fn elec_state(&self) -> Option<&SimElectronicState> {
        self.elec_state.as_ref()
    }

    pub fn cfg(&self) -> &SimConfig {
        &self.cfg
    }

    pub fn step(&mut self, energy_level: usize) {
        self.recalculate();

        let dt = 10.0;
        let art = self.elec_state.as_ref().unwrap();
        for nucleus in &mut self.state.nuclei {
            let psi = &art.eigenstates[energy_level];

            if let Some((x, y)) = bounds_check(nucleus.pos.x.round() as i32, nucleus.pos.y.round() as i32, psi) {
                let force = interpolate_force_vector(&art, energy_level, nucleus.pos);

                nucleus.vel += force * dt / NUCLEAR_MASS;
            } else {
                if nucleus.pos.x < 0.0 || nucleus.pos.x + 1.0 > psi.ncols() as f32 {
                    nucleus.vel.x *= -1.0;
                }

                if nucleus.pos.y < 0.0 || nucleus.pos.y + 1.0 > psi.nrows() as f32 {
                    nucleus.vel.y *= -1.0;
                }
            }

            nucleus.pos += nucleus.vel * dt;
        }
    }
}

pub fn interpolate_force_vector(art: &SimElectronicState, energy_level: usize, pos: Vec2) -> Vec2 {
    let tl_x = pos.x as i32;
    let tl_y = pos.y as i32;

    let xf = pos.x.fract();
    let yf = pos.y.fract();

    let parts = [
        (0, 0, 1. - xf, 1. - yf),
        (1, 0, xf, 1. - yf),
        (0, 1, 1. - xf, yf),
        (1, 1, xf, yf),
    ];

    // Accumulate samples into adjacent points
    let mut sum = Vec2::ZERO;
    for (off_x, off_y, interp_x, interp_y) in parts {
        let psi = &art.eigenstates[energy_level];
        if let Some((x, y)) = bounds_check(tl_x + off_x, tl_y + off_y, psi) {
            sum += interp_x * interp_y * compute_force_at(art, energy_level, x, y);
        }
    }

    sum
}

fn calculate_electric_state(
    cfg: &SimConfig,
    state: &SimState,
    cache: Option<Cache>,
) -> (SimElectronicState, Cache) {
    let potential = match cfg.potental_mode {
        PotentialMode::Delta => calculate_delta_potential(cfg, state),
        PotentialMode::Kqr => calculate_potential_r_squared(cfg, state),
    };

    let ham = HamiltonianObject::from_potential(potential, cfg);
    let (mut energies, mut eigenstates, mut cache) = solve_schrödinger(cfg, &ham, cache);

    /*
    // THIS IS A WORKAROUND TO A KNOWN BEHAVIOUR EFFECTING CONVERGENCE 
    // If there's at least one negative potential, we should have a bound state. 
    // If this is not the case, then we should try re-calculating without the preconditioner.
    // This costs a lot of CPU cycles but ensures convergence
    if cfg.v0 < 0. && state.nuclei.len() >= 1 && energies[0] > 0. {
        eprintln!("Positive first energy refresh triggered");
        (energies, eigenstates, cache) = solve_schrödinger(cfg, &ham, None);
    }
    */

    (
        SimElectronicState {
            eigenstates,
            energies,
            ham,
        },
        cache,
    )
}

/// Calculate the electric potential in the position basis
fn calculate_delta_potential(cfg: &SimConfig, state: &SimState) -> Grid2D<f32> {
    let mut pot = Grid2D::zeros((cfg.grid_width, cfg.grid_width));
    for nucleus in &state.nuclei {
        if let Some(grid_coord) = bounds_check(nucleus.pos.x.round() as i32, nucleus.pos.y.round() as i32, &pot) {
            pot[grid_coord] += cfg.v0;
        }
    }

    pot
}

/// Calculate the electric potential in the position basis
fn calculate_potential_r_squared(cfg: &SimConfig, state: &SimState) -> Grid2D<f32> {
    let mut potential = Grid2D::zeros((cfg.grid_width, cfg.grid_width));

    for nucleus in &state.nuclei {
        interp_write(&mut potential, nucleus.pos.x, nucleus.pos.y, cfg.v0);
    }

    potential
}

fn interp_write(grid: &mut Grid2D<f32>, x: f32, y: f32, value: f32) {
    let tl_x = x as i32;
    let tl_y = y as i32;

    let xf = x.fract();
    let yf = y.fract();

    let parts = [
        (0, 0, 1. - xf, 1. - yf),
        (1, 0, xf, 1. - yf),
        (0, 1, 1. - xf, yf),
        (1, 1, xf, yf),
    ];

    // Accumulate samples into adjacent points
    for (off_x, off_y, interp_x, interp_y) in parts {
        if let Some(grid_pos) = bounds_check(tl_x + off_x, tl_y + off_y, &grid) {
            grid[grid_pos] += interp_x * interp_y * value;
        }
    }
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
fn bounds_check<T>(pt_x: i32, pt_y: i32, arr: &Array2<T>) -> Option<(usize, usize)> {
    (pt_x >= 0 && pt_y >= 0 && pt_x < arr.ncols() as i32 && pt_y < arr.nrows() as i32)
        .then(|| (pt_x as usize, pt_y as usize))
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
    pub potential: Grid2D<f32>,
    cfg: SimConfig,
}

impl HamiltonianObject {
    pub fn from_potential(potential: Grid2D<f32>, cfg: &SimConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            // Diagonal includes both the potential AND the stencil centers
            potential,
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

    pub fn value_at(&self, x: usize, y: usize, psi: &Grid2D<f32>) -> f32 {
        let center_grid_coord = (x, y);

        let mut sum = 0.0;

        let pot = self.potential[center_grid_coord];

        let h2m = -HBAR.powi(2) / 2. / ELECTRON_MASS;

        for (off, coeff) in [
            ((-1, 0), 1.0 * h2m),
            ((1, 0), 1.0 * h2m),
            ((0, 1), 1.0 * h2m),
            ((0, -1), 1.0 * h2m),
            ((0, 0), -4.0 * h2m + pot),
        ] {
            if let Some(grid_coord) = bounds_check(x as i32 + off.0, y as i32 + off.1, &psi) {
                sum += coeff * psi[grid_coord];
            }
        }

        sum
    }

    pub fn matrix_vector_prod(&self, psi: Grid2D<f32>) -> Grid2D<f32> {
        let mut output = Grid2D::zeros((self.cfg.grid_width, self.cfg.grid_width));

        for y in 0..psi.nrows() {
            for x in 0..psi.ncols() {
                output[(x, y)] = self.value_at(x, y, &psi);
            }
        }

        output
    }

    // NOTE: This operation is not in the hot path so it is NOT optimized!
    pub fn matrix_matrix_prod(&self, mut mtx: Array2<f32>) -> Array2<f32> {
        for mut column in mtx.columns_mut() {
            let res = self.matrix_vector_prod(vector_to_state(column.to_owned(), &self.cfg));
            column.assign(&state_to_vector(res));
        }
        mtx
    }
}


pub fn gradient_at(x: usize, y: usize, psi: &Grid2D<f32>) -> Vec2 {
    let mut sum_x: f32 = 0.;
    let mut sum_y: f32 = 0.;

    // Five-point stencil https://en.wikipedia.org/wiki/Five-point_stencil
    for (offset, coefficient) in (-2..=2).zip(&[1. / 12., -8./12., 0., 8./12., -1. / 12.]) {
        if let Some(grid_coord) = bounds_check(x as i32 + offset, y as i32, &psi) {
            sum_x += psi[grid_coord] * coefficient;
        }

        if let Some(grid_coord) = bounds_check(x as i32, y as i32 + offset, &psi) {
            sum_y += psi[grid_coord] * coefficient;
        }
    }

    Vec2::new(sum_x, sum_y)
}

pub fn compute_force_at(art: &SimElectronicState, energy_level: usize, x: usize, y: usize) -> Vec2 {
    let psi = &art.eigenstates[energy_level];
    let grad_psi = gradient_at(x, y, psi);
    let energy = art.energies[energy_level];

    // grad * H * psi = E * grad * psi

    -psi[(x, y)] * energy * grad_psi
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
    ham: &HamiltonianObject,
    cache: Option<Cache>,
) -> (Vec<f32>, Vec<Grid2D<f32>>, Cache) {
    let cache = cache.filter(|p| p.shape()[0] == ham.ncols());
    let cache = cache.filter(|p| p.shape()[1] == cfg.n_states);

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
        cfg.eigval_search,
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
                .map(|col| vector_to_state(col.to_owned(), cfg))
                .collect();

            cache = eig.eigvecs;
        }
        LobpcgResult::Err((e, None)) => panic!("{}", e),
    }

    (eigvals, eigvects, cache)
}

impl Nucleus {
    pub fn stationary_at(pos: Vec2) -> Self {
        Self { vel: Vec2::ZERO, pos }
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

fn state_to_vector(state: Grid2D<f32>) -> Array1<f32> {
    let num_elem = state.nrows() * state.ncols();
    state.into_shape(num_elem).unwrap()
}

fn vector_to_state(state: Array1<f32>, cfg: &SimConfig) -> Grid2D<f32> {
    state.into_shape((cfg.grid_width, cfg.grid_width)).unwrap()
}
