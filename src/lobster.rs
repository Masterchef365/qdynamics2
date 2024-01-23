use crate::{
    array2d::Array2D,
    sim::{
        linear_downscale_by_two, nearest_upscale_by_two, solve_schrödinger, EigenAlgorithm,
        HamiltonianObject, SimArtefacts,
    },
};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

const MIN_GRID_WIDTH: usize = 10;

pub fn lobster(ham: HamiltonianObject) -> (Vec<f32>, Vec<Array2D<f32>>) {
    let mut cache = None;
    let mut smaller_cfg = ham.cfg().clone();
    if smaller_cfg.grid_width > MIN_GRID_WIDTH {
        smaller_cfg.grid_width /= 2;
        let smaller_potential = linear_downscale_by_two(ham.potential());
        smaller_cfg.eig_algo = EigenAlgorithm::Lobster;

        let (precond_energies, precond_eigvects) =
            solve_schrödinger(&smaller_cfg, &smaller_potential, None);
        dbg!(smaller_cfg.grid_width);

        cache = Some(SimArtefacts {
            eigenstates: precond_eigvects
                .into_iter()
                .map(|v| nearest_upscale_by_two(&v))
                .collect(),
            energies: precond_energies,
            potential: ham.potential().clone(),
        });
    }

    let mut big_cfg = ham.cfg().clone();
    big_cfg.eig_algo = EigenAlgorithm::LobPcg;
    solve_schrödinger(&big_cfg, ham.potential(), cache)
}
