use nalgebra::{DMatrix, MatrixN, Point2, Vector2};

use crate::array2d::Array2D;

const NUCLEAR_MASS: f32 = 1.0;
const ELECTRON_MASS: f32 = 1.0;
const HBAR: f32 = 1.0;

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
    let (energies, eigenstates) = calculate_energy_eigenbasis(cfg, &potential);

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

fn calculate_energy_eigenbasis(
    cfg: &SimConfig,
    potential: &Array2D<f32>,
) -> (Vec<f32>, Vec<Array2D<f32>>) {
    todo!()
}

/// Calculates the array index of a
fn calc_grid_array_index(x: i32, y: i32, width: i32) -> Option<usize> {
    (x > 0 && y > 0 && x < width && y < width).then(|| (x + y * width) as usize)
}

/// Generates the second-derivative finite-difference stencil in the position basis
/// NOTE: This assumes the function is zero outside of the boundaries!
fn second_derivative_matrix(width: i32) -> DMatrix<f32> {
    let n = width * width;
    // Oh dear this matrix is going to be fucking HUGE
    let mut mat = DMatrix::zeros(n, n);

    for x in 0..width {
        for y in 0..width {
            for (off_x, off_y, coefficient) in [
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
                (0, 0, -4.0),
            ] {
                mat[calc_grid_array_index(x+off_x, y+off_y, width)] += coefficient;
            }
        }
    }
}

fn calculate_classical_energy(cfg: &SimConfig, state: &SimState) -> f32 {
    let kinetic_energy: f32 = state
        .nuclei
        .iter()
        .map(|nucleus| NUCLEAR_MASS * nucleus.vel.magnitude_squared() / 2.)
        .sum();

    todo!()
}
