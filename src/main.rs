use egui::{CentralPanel, DragValue, SelectableLabel, SidePanel, Ui, Response};
use image_view::{array_to_imagedata, ImageViewWidget};
//#![warn(clippy::all, rust_2018_idioms)]
//#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
//
use nalgebra::{Point2, Vector2};
use qdynamics::sim::{EigenAlgorithm, Nucleus, Sim, SimConfig, SimState, SimArtefacts};

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
    edit_cfg: SimConfig,
    edit_initial_state: SimState,
    sim: Sim,
    img: ImageViewWidget,
    viewed_eigstate: usize,
    show_probability: bool,
    max_states_is_grid_width: bool,
    cache: Option<SimArtefacts>,
}

impl Default for TemplateApp {
    fn default() -> Self {
        let cfg = initial_cfg();
        let edit_state = initial_state(&cfg);

        let sim = Sim::new(cfg.clone(), edit_state.clone(), None);
        let img = ImageViewWidget::default();

        Self {
            cache: None,
            edit_initial_state: edit_state,
            sim,
            img,
            viewed_eigstate: 0,
            edit_cfg: cfg,
            show_probability: false,
            max_states_is_grid_width: false,
        }
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

        let mut needs_update = false;
        let mut needs_recalculate = false;

        if self.max_states_is_grid_width {
            if self.edit_cfg.n_states != self.edit_cfg.grid_width {
                self.edit_cfg.n_states = self.edit_cfg.grid_width;
                needs_recalculate = true;
            }
        }

        SidePanel::left("left_panel").show(ctx, |ui| {
            ui.strong("View");
            let energies = &self.sim.artefacts().energies;
            needs_update |= ui
                .add(DragValue::new(&mut self.viewed_eigstate).clamp_range(0..=energies.len() - 1))
                .changed();
            ui.label(format!("Energy: {}", energies[self.viewed_eigstate]));
            needs_update |= ui
                .checkbox(&mut self.show_probability, "Show probability")
                .changed();

            ui.separator();
            ui.strong("Config");
            needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.edit_cfg.v0)
                        .speed(1e-1)
                        .prefix("V0: "),
                )
                .changed();

            needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.edit_cfg.grid_width)
                        .speed(1e-1)
                        .prefix("Grid size: "),
                )
                .changed();

            needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.edit_cfg.tolerance)
                        .speed(1e-1)
                        .prefix("Tolerance: "),
                )
                .changed();

            needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.edit_cfg.num_solver_iters)
                        .speed(1e-1)
                        .prefix("Max iters: "),
                )
                .changed();

            ui.horizontal(|ui| {
                needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.edit_cfg.n_states)
                        .speed(1e-1)
                        .prefix("Max states: "),
                )
                .changed();

                ui.checkbox(&mut self.max_states_is_grid_width, "From width");
            });

            ui.horizontal(|ui| {
                needs_recalculate |= ui
                    .selectable_value(
                        &mut self.edit_cfg.eig_algo,
                        EigenAlgorithm::Lanczos,
                        "Lanczos",
                    )
                    .clicked();
                needs_recalculate |= ui
                    .selectable_value(
                        &mut self.edit_cfg.eig_algo,
                        EigenAlgorithm::LobPcg,
                        "LobPcg",
                    )
                    .clicked();
                needs_recalculate |= ui
                    .selectable_value(
                        &mut self.edit_cfg.eig_algo,
                        EigenAlgorithm::Lobster,
                        "Lobster",
                    )
                    .clicked();
                needs_recalculate |= ui
                    .selectable_value(
                        &mut self.edit_cfg.eig_algo,
                        EigenAlgorithm::Nalgebra,
                        "Nalgebra",
                    )
                    .clicked();
            });


            ui.separator();
            ui.strong("Nuclei");
            needs_recalculate |= sim_state_editor(ui, &mut self.edit_initial_state);
        });

        if needs_recalculate {
            self.recalculate(ctx);
        } else if needs_update {
            self.update_view(ctx);
        }

        CentralPanel::default().show(ctx, |ui| {
            self.img.show(ui);
        });
    }
}

impl TemplateApp {
    fn recalculate(&mut self, ctx: &egui::Context) {
        self.sim = Sim::new(self.edit_cfg.clone(), self.edit_initial_state.clone(), self.cache.take());
        self.cache = Some(self.sim.artefacts().clone());
        self.update_view(ctx);
    }

    fn update_view(&mut self, ctx: &egui::Context) {
        self.viewed_eigstate = self
            .viewed_eigstate
            .min(self.sim.artefacts().energies.len() - 1);
        let eigstate = &self.sim.artefacts().eigenstates[self.viewed_eigstate];

        let w = eigstate.data().len();

        let image;
        if self.show_probability {
            let sum: f32 = eigstate.data().iter().map(|v| v.powi(2)).sum();
            image = eigstate.map(|v| {
                let v = (v.powi(2) / sum) as f32;
                let v = 100.0 * v;
                [v, v, v, 0.0]
            });
        } else {
            image = eigstate.map(|v| {
                let v = *v as f32 * (w as f32).sqrt();
                if v > 0. {
                    [v, 0.3 * v, 0.0, 0.0]
                } else {
                    [0.0, 0.3 * -v, -v, 0.0]
                }
            });
        }
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
            pos: Point2::new(cfg.grid_width as f32 / 2., cfg.grid_width as f32 / 2.),
            vel: Vector2::zeros(),
        }],
    }
}

const N: usize = 2_usize.pow(7);
fn initial_cfg() -> SimConfig {
    SimConfig {
        dx: 1.0,
        grid_width: N,
        v0: -1.,
        v_soft: 0.1,
        v_scale: 1.,
        n_states: 30,
        num_solver_iters: 10,
        eig_algo: EigenAlgorithm::Lobster,
        tolerance: 1e-2,
    }
}

fn sim_state_editor(ui: &mut Ui, state: &mut SimState) -> bool {
    let mut needs_recalculate = false;

    // Just nuclei for now
    let mut delete = None;
    for (idx, nucleus) in state.nuclei.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            needs_recalculate |= ui.add(DragValue::new(&mut nucleus.pos.x).prefix("x: ").speed(1e-1)).changed();
            needs_recalculate |= ui.add(DragValue::new(&mut nucleus.pos.y).prefix("y: ").speed(1e-1)).changed();

            if ui.button("Delete").clicked() {
                delete = Some(idx);
                needs_recalculate = true;
            }
        });
    }

    if let Some(idx) = delete {
        state.nuclei.remove(idx);
        needs_recalculate = true;
    }

    if ui.button("Add").clicked() {
        state.nuclei.push(Nucleus::default());
        needs_recalculate = true;
    }

    needs_recalculate
}
