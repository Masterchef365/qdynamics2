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

/// Returns false if out of bounds with the given width
fn bounds_check(pt: Point2<i32>, width: i32) -> bool {
    pt.x > 0 && pt.y > 0 && pt.x < width && pt.y < width
}

/// Solves the Schrödinger equation for the first N energy eigenstates
///
/// Generates the second-derivative finite-difference stencil in the position basis. This is then
/// combined with the potential to form the Hamiltonian.
/*
///
///                 H = (-ħ/2m)∇² + V
///
/// In the 2D finite difference the stencil looks like:
///                   | 0   1   0 |
///                   | 1  V-4  1 |
///                   | 0   1   0 |
*/
fn solve_schrödinger(cfg: &SimConfig, potential: &Array2D<f32>) -> (Vec<f32>, Vec<Array2D<f32>>) {
    assert_eq!(cfg.grid_width, potential.width());

    // Width
    let w = cfg.grid_width as i32;

    let av = |input_vector, mut output_vector| {
        ()
    };

    let vector_length = potential.width() * potential.height();

    // https://gitlab.com/solidtux-rust/arpack-ng/-/blob/master/examples/simple.rs?ref_type=heads
    // https://help.scilab.org/docs/5.3.1/en_US/znaupd.html
    // https://docs.rs/arpack-ng/latest/src/arpack_ng/ndarray.rs.html#9-57

    arpack_ng::eigenvectors(
        av,
        vector_length,
        &arpack_ng::Which::SmallestRealPart,
        cfg.n_states,
        vector_length,
        cfg.num_solver_iters,
    );

    todo!()
}

fn calculate_classical_energy(cfg: &SimConfig, state: &SimState) -> f32 {
    let kinetic_energy: f32 = state
        .nuclei
        .iter()
        .map(|nucleus| NUCLEAR_MASS * nucleus.vel.magnitude_squared() / 2.)
        .sum();

    todo!()
}
