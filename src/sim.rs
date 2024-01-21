use std::time::Instant;

use eigenvalues::matrix_operations::MatrixOperations;
use nalgebra::{DMatrix, MatrixN, Point2, Vector2, ComplexField, DVector};
use num_complex::Complex64;

use crate::array2d::Array2D;

const NUCLEAR_MASS: f32 = 1.0;
const ELECTRON_MASS: f32 = 1.0;
const HBAR: f32 = 1.0;

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
}

pub struct Nucleus {
    // pub mass: f32,
    /// Velocity
    pub vel: Vector2<f32>,
    /// Position
    pub pos: Point2<f32>,
}

pub struct SimState {
    /// Index of the quantum energy level (lambda)
    pub energy_level: usize,
    /// Wavefunction coefficients (c_j)
    pub coeffs: Vec<f32>,
    /// Atomic nuclei (R, P)
    pub nuclei: Vec<Nucleus>,
}

/// Data uniquely generated from a SimState
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
    pub cfg: SimConfig,
    pub state: SimState,
    pub artefacts: SimArtefacts,
}

impl Sim {
    pub fn new(cfg: SimConfig, state: SimState) -> Self {
        let artefacts = calculate_artefacts(&cfg, &state);

        Self {
            state,
            cfg,
            artefacts,
        }
    }
}

fn calculate_artefacts(cfg: &SimConfig, state: &SimState) -> SimArtefacts {
    let potential = calculate_potential(cfg, state);
    let (energies, eigenstates) = solve_schrödinger(cfg, &potential);

    let psi = eigenstates[0].map(|v| Complex64::from(*v as f64));
    let h_psi = hamiltonian(cfg, &psi, &potential);
    let percent_error: Vec<f64> = h_psi
        .data()
        .iter()
        .zip(psi.data())
        .zip(&energies)
        .map(|((hpsi, psi), energy)| {
            (hpsi - *energy as f64 * psi).abs() / (*energy as f64 * psi).abs()
        })
        .collect();

    dbg!(percent_error);
    panic!();

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
    let mut output = Array2D::new(cfg.grid_width, cfg.grid_width);

    for y in 0..cfg.grid_width {
        for x in 0..cfg.grid_width {
            output[(x, y)] = Point2::new(x as f32, y as f32) * cfg.dx;
        }
    }

    output
}

/// Returns false if out of bounds with the given width
fn bounds_check(pt: Point2<i32>, width: i32) -> Option<(usize, usize)> {
    (pt.x > 0 && pt.y > 0 && pt.x < width && pt.y < width).then(|| (pt.x as usize, pt.y as usize))
}

///
///                 H = (-ħ/2m)∇² + V
///
/// In the 2D finite difference the stencil looks like:
///                   | 0   k   0 |
///                   | k  V-4k k | where k = -ħ/2m
///                   | 0   k   0 |
fn hamiltonian(
    cfg: &SimConfig,
    psi: &Array2D<f32>,
    potential: &Array2D<f32>,
) -> Array2D<f32> {
    let mut output = Array2D::new(psi.width(), psi.height());

    for x in 0..psi.width() {
        for y in 0..psi.height() {
            let center_world_coord = Point2::new(x as i32, y as i32);
            let center_grid_coord = (x, y);

            let potential_pt = potential[center_grid_coord];

            let mut sum = 0.0;

            for (off, coefficient) in [
                (Vector2::new(-1, 0), 1.0),
                (Vector2::new(1, 0), 1.0),
                (Vector2::new(0, 1), 1.0),
                (Vector2::new(0, -1), 1.0),
                (Vector2::new(0, 0), -4.0),
            ] {
                if let Some(grid_coord) = bounds_check(center_world_coord + off, psi.width() as i32)
                {
                    sum += coefficient * psi[grid_coord];
                }
            }

            let kinetic = sum * (-HBAR / ELECTRON_MASS / 2.0 / cfg.dx.powi(2));

            output[center_grid_coord] = kinetic + potential_pt;
        }
    }

    output
}

#[derive(Clone)]
struct HamiltonianObject {
    diag: DVector<f64>,
    cfg: SimConfig,
}

impl HamiltonianObject {
    pub fn from_potential(potential: &Array2D<f32>, cfg: &SimConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            // Diagonal includes both the potential AND the stencil centers
            diag: potential.data().iter().map(|v| *v as f64 - 4.0).collect(),
        }
    }
}

impl MatrixOperations for HamiltonianObject {
    fn ncols(&self) -> usize {
        self.diag.data().len()
    }

    fn nrows(&self) -> usize {
        self.diag.data().len()
    }

    fn diagonal(&self) -> DVector<f64> {
        
    }
}

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

/// Solves the Schrödinger equation for the first N energy eigenstates
///
/// Generates the second-derivative finite-difference stencil in the position basis. This is then
/// combined with the potential to form the Hamiltonian.
fn solve_schrödinger(cfg: &SimConfig, potential: &Array2D<f32>) -> (Vec<f32>, Vec<Array2D<f32>>) {
    assert_eq!(cfg.grid_width, potential.width());

    // Width
    let vector_length = potential.width() * potential.height();

    // https://gitlab.com/solidtux-rust/arpack-ng/-/blob/master/examples/simple.rs?ref_type=heads
    // https://help.scilab.org/docs/5.3.1/en_US/znaupd.html
    // https://docs.rs/arpack-ng/latest/src/closure/closure.rs.html#6-17
    // https://docs.rs/arpack-ng/latest/src/arpack_ng/ndarray.rs.html#9-57

    let start = Instant::now();
    let (energies, eigenstates) = arpack_ng::eigenvectors(
        |input_vector, mut output_vector| {
            hamiltonian_flat(
                cfg,
                potential,
                input_vector.as_slice().unwrap(),
                output_vector.as_slice_mut().unwrap(),
            );
        },
        vector_length,
        &arpack_ng::Which::SmallestRealPart,
        cfg.n_states,
        vector_length,
        cfg.num_solver_iters,
    )
    .unwrap();
    let time = start.elapsed().as_secs_f32();

    // Ensure there is no complex component of energy nor eigenstate
    dbg!(&eigenstates);
    dbg!(&energies);

    dbg!(time);
    dbg!(eigenstates.shape());

    // Yeesh
    let thresh = 1.0;
    assert!(energies.iter().all(|energy| energy.im.abs() < thresh));
    assert!(eigenstates.iter().all(|entry| entry.im.abs() < thresh));

    let energies: Vec<f32> = energies.iter().map(|energy| energy.re as f32).collect();

    // TODO: Is outer_iter really the right one?
    let eigenstates: Vec<Array2D<f32>> = eigenstates
        .axis_iter(ndarray::Axis(1))
        .map(|eigenstate| {
            let eigenstate: Vec<f32> = eigenstate.iter().map(|entry| entry.re as f32).collect();
            Array2D::from_array(cfg.grid_width, eigenstate)
        })
        .collect();

    (energies, eigenstates)
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

impl Sim {
    pub fn step(&mut self) {}
}
