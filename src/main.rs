use egui::{CentralPanel, SidePanel, DragValue};
use image_view::{ImageViewWidget, array_to_imagedata};
//#![warn(clippy::all, rust_2018_idioms)]
//#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
//
use nalgebra::{Point2, Vector2};
use qdynamics::sim::{Nucleus, Sim, SimConfig, SimState};

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                // NOE: Adding an icon is optional
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .unwrap(),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "eframe template",
        native_options,
        Box::new(|cc| Box::new(TemplateApp::new(cc))),
    )
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn main() {
    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        eframe::WebRunner::new()
            .start(
                "the_canvas_id", // hardcode it
                web_options,
                Box::new(|cc| Box::new(TemplateApp::new(cc))),
            )
            .await
            .expect("failed to start eframe");
    });
}

mod image_view;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
pub struct TemplateApp {
    sim: Sim,
    img: ImageViewWidget,
    viewed_eigstate: usize,
}

impl Default for TemplateApp {
    fn default() -> Self {
        let cfg = initial_cfg();
        let state = initial_state(&cfg);

        let sim = Sim::new(cfg, state);
        let img = ImageViewWidget::default();

        Self { sim, img, viewed_eigstate: 0 }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut inst = Self::default();

        inst.update_view(&cc.egui_ctx);

        inst
    }
}

impl eframe::App for TemplateApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.sim.step();

        CentralPanel::default().show(ctx, |ui| {
            self.img.show(ui);
        });

        SidePanel::left("left_panel").show(ctx, |ui| {
            let energies = &self.sim.artefacts.energies;
            let res = ui.add(DragValue::new(&mut self.viewed_eigstate).clamp_range(0..=energies.len()-1));
            ui.label(format!("Energy: {}", energies[self.viewed_eigstate]));

            if res.changed() {
                self.update_view(ctx);
            }
        });
    }
}

impl TemplateApp {
    fn update_view(&mut self, ctx: &egui::Context) {
        let eigstate = &self.sim.artefacts.eigenstates[self.viewed_eigstate];

        let w = eigstate.data().len();
        let image = eigstate.map(|v| {
            let v = *v as f32 * (w as f32).sqrt();
            if v > 0. {
                [v, 0.1 * v, 0.0, 0.0]
            } else {
                [0.0, 0.1 * v, v, 0.0]
            }
        });
        let image = array_to_imagedata(&image);

        self.img.set_image("Spronkus".into(), ctx, image);

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

const N: usize = 20;
fn initial_cfg() -> SimConfig {
    SimConfig {
        dx: 1.0,
        grid_width: N,
        v0: -1.,
        v_soft: 0.1,
        v_scale: 1.,
        n_states: 3,
        num_solver_iters: 100,
    }
}
