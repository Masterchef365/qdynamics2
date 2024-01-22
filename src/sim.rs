use std::time::Instant;

use eigenvalues::{
    davidson::Davidson, lanczos::HermitianLanczos, matrix_operations::MatrixOperations,
    DavidsonCorrection, SpectrumTarget,
};
use nalgebra::{
    ComplexField, DMatrix, DMatrixSlice, DVector, DVectorSlice, MatrixN, Point2, Vector2,
};
use num_complex::Complex64;

use crate::array2d::Array2D;

// TODO: Set these parameters ...
const NUCLEAR_MASS: f64 = 1.0;
const ELECTRON_MASS: f64 = 1.0;
const HBAR: f64 = 1.0;

#[derive(Clone)]
pub struct SimConfig {
    // Options which should NOT change during runtime
    /// Spacing between adjacent points on the grid
    pub dx: f64,
    /// Grid width
    pub grid_width: usize,

    // Options which should may be able to change during runtime...
    /// Depth of potential wells due to nuclei
    pub v0: f64,
    /// Potential function softening factor
    pub v_soft: f64,
    /// Potential function scale factor
    pub v_scale: f64,

    /// Number of eigenstates needed
    pub n_states: usize,
    pub num_solver_iters: usize,
    // /// Mass of each nucleus
}

pub struct Nucleus {
    // pub mass: f64,
    /// Velocity
    pub vel: Vector2<f64>,
    /// Position
    pub pos: Point2<f64>,
}

pub struct SimState {
    /// Index of the quantum energy level (lambda)
    pub energy_level: usize,
    /// Wavefunction coefficients (c_j)
    pub coeffs: Vec<f64>,
    /// Atomic nuclei (R, P)
    pub nuclei: Vec<Nucleus>,
}

/// Data uniquely generated from a SimState
pub struct SimArtefacts {
    /// Energy eigenstates (psi_n)
    pub eigenstates: Vec<Array2D<f64>>,
    /// Energy levels (E_n)
    pub energies: Vec<f64>,
    /// Map of electric potential due to nuclei
    pub potential: Array2D<f64>,
    // /// Amount of energy in the classical system at present time
    // pub classical_energy: f64,
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

    /*
    let psi = eigenstates[0].map(|v| *v as f64);
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
    */
    todo!()
}

/// Calculate the electric potential in the position basis
fn calculate_potential(cfg: &SimConfig, state: &SimState) -> Array2D<f64> {
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

fn softened_potential(r: f64, cfg: &SimConfig) -> f64 {
    cfg.v0 / (r * cfg.v_scale + cfg.v_soft)
}

/// World-space positions at grid points
fn grid_positions(cfg: &SimConfig) -> Array2D<Point2<f64>> {
    let mut output = Array2D::from_array(
        cfg.grid_width,
        vec![Point2::origin(); cfg.grid_width.pow(2)],
    );

    for y in 0..cfg.grid_width {
        for x in 0..cfg.grid_width {
            output[(x, y)] = Point2::new(x as f64, y as f64) * cfg.dx;
        }
    }

    output
}

/// Returns false if out of bounds with the given width
fn bounds_check(pt: Point2<i32>, width: i32) -> Option<(usize, usize)> {
    Some((
        pt.x.rem_euclid(width) as usize,
        pt.y.rem_euclid(width) as usize,
    ))
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
    potential: Array2D<f64>,
    cfg: SimConfig,
}

impl HamiltonianObject {
    pub fn from_potential(potential: &Array2D<f64>, cfg: &SimConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            // Diagonal includes both the potential AND the stencil centers
            potential: potential.clone(),
        }
    }
}

fn nalgebra_to_array2d(vs: DVectorSlice<f64>, cfg: &SimConfig) -> Array2D<f64> {
    Array2D::from_array(cfg.grid_width, vs.as_slice().to_vec())
}

fn array2d_to_nalgebra(arr: &Array2D<f64>) -> DVector<f64> {
    arr.data().to_vec().into()
}

impl MatrixOperations for HamiltonianObject {
    fn ncols(&self) -> usize {
        self.potential.data().len()
    }

    fn nrows(&self) -> usize {
        self.potential.data().len()
    }

    fn diagonal(&self) -> DVector<f64> {
        unimplemented!()
    }

    fn set_diagonal(&mut self, _diag: &DVector<f64>) {
        unimplemented!()
    }

    fn matrix_vector_prod(&self, vs: DVectorSlice<f64>) -> DVector<f64> {
        let psi = nalgebra_to_array2d(vs, &self.cfg);

        let mut output = Array2D::new(self.cfg.grid_width, self.cfg.grid_width);

        for x in 0..psi.width() {
            for y in 0..psi.height() {
                let center_world_coord = Point2::new(x as i32, y as i32);
                let center_grid_coord = (x, y);

                let mut sum = 0.0;

                for (off, coeff) in [
                    (Vector2::new(-1, 0), 1.0),
                    (Vector2::new(1, 0), 1.0),
                    (Vector2::new(0, 1), 1.0),
                    (Vector2::new(0, -1), 1.0),
                    (Vector2::new(0, 0), self.potential[center_grid_coord] - 4.0),
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
    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<f64>) -> DMatrix<f64> {
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
potential: &Array2D<f64>,
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
fn solve_schrödinger(cfg: &SimConfig, potential: &Array2D<f64>) -> (Vec<f64>, Vec<Array2D<f64>>) {
    assert_eq!(cfg.grid_width, potential.width());

    let start = Instant::now();
    let eig = HermitianLanczos::new(
        HamiltonianObject::from_potential(potential, cfg),
        cfg.n_states,
        SpectrumTarget::Lowest,
    )
    .unwrap();
    let time = start.elapsed().as_secs_f64();

    let sel_idx = 0;

    // Calculate selected eigenstate and energy
    let sel_energy = eig.eigenvalues[sel_idx];
    let sel_energy_eigenstate = eig.eigenvectors.column(sel_idx);
    let hpsi = HamiltonianObject::from_potential(potential, cfg)
        .matrix_vector_prod(sel_energy_eigenstate);

    let expect = sel_energy * sel_energy_eigenstate;

    let percent_err = (hpsi - &expect).abs().component_div(&expect.abs());
    let avg_err = percent_err.sum() / percent_err.len() as f64;

    dbg!(percent_err);
    dbg!(avg_err);
    dbg!(&eig.eigenvalues[sel_idx]);

    // Ensure there is no complex component of energy nor eigenstate
    //dbg!(&eig.eigenvalues);
    //dbg!(&eig.eigenvectors);

    //dbg!(time);
    /*
    dbg!(eigenstates.shape());

    // Yeesh
    let thresh = 1.0;
    assert!(energies.iter().all(|energy| energy.im.abs() < thresh));
    assert!(eigenstates.iter().all(|entry| entry.im.abs() < thresh));

    let energies: Vec<f64> = energies.iter().map(|energy| energy.re as f64).collect();

    // TODO: Is outer_iter really the right one?
    let eigenstates: Vec<Array2D<f64>> = eigenstates
    .axis_iter(ndarray::Axis(1))
    .map(|eigenstate| {
    let eigenstate: Vec<f64> = eigenstate.iter().map(|entry| entry.re as f64).collect();
    Array2D::from_array(cfg.grid_width, eigenstate)
    })
    .collect();

    (energies, eigenstates)
    */
    todo!()
}

/*
fn calculate_classical_energy(cfg: &SimConfig, state: &SimState) -> f64 {
let kinetic_energy: f64 = state
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
