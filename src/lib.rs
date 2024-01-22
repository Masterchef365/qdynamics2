use nalgebra::{Point2, Vector2};
use sim::{Nucleus, Sim, SimConfig, SimState};

mod array2d;
mod sim;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
pub struct TemplateApp {
    sim: Sim,
}

impl Default for TemplateApp {
    fn default() -> Self {
        let cfg = initial_cfg();
        let state = initial_state(&cfg);
        Self {
            sim: Sim::new(cfg, state),
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Default::default()
    }
}

impl eframe::App for TemplateApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.sim.step();
    }
}

fn initial_state(cfg: &SimConfig) -> SimState {
    SimState {
        energy_level: 0,
        coeffs: (0..cfg.grid_width.pow(2))
            .map(|n| if n == 0 { 1.0 } else { 0.0 })
            .collect(),
        nuclei: vec![Nucleus {
            pos: Point2::new(cfg.grid_width as f64 / 2., cfg.grid_width as f64 / 2.),
            vel: Vector2::zeros(),
        }],
    }
}

const N: usize = 5;
fn initial_cfg() -> SimConfig {
    SimConfig {
        dx: 1.0,
        grid_width: N,
        v0: -1.,
        v_soft: 0.1,
        v_scale: 1.,
        n_states: N.pow(2),
        num_solver_iters: 100,
    }
}
