use egui::{CentralPanel, DragValue, Response, SelectableLabel, SidePanel, Ui};
use glam::Vec2;
//#![warn(clippy::all, rust_2018_idioms)]
//#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
//
use qdynamics::sim::{Nucleus, PotentialMode, Sim, SimArtefacts, SimConfig, SimState};
use widgets::{
    display_imagedata, electric_editor, nucleus_editor, ImageViewWidget, StateViewConfig,
};

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

mod widgets;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
pub struct TemplateApp {
    sim: Sim,
    img: ImageViewWidget,
    view_cfg: StateViewConfig,
}

impl Default for TemplateApp {
    fn default() -> Self {
        let cfg = initial_cfg();
        let edit_state = initial_state(&cfg);

        let sim = Sim::new(cfg, edit_state);
        let img = ImageViewWidget::default();

        Self {
            sim,
            img,
            view_cfg: StateViewConfig::default(),
            //max_states_is_grid_width: true,
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

        /*
        if self.max_states_is_grid_width {
            if self.sim.cfg.n_states != self.sim.cfg.grid_width {
                self.sim.cfg.n_states = self.sim.cfg.grid_width;
                needs_recalculate = true;
            }
        }
        */

        SidePanel::left("left_panel").show(ctx, |ui| {
            if let Some(artefacts) = self.sim.artefacts() {
                ui.strong("Viewed eigenstate:");
                let energies = &artefacts.energies;
                needs_update |= ui
                    .add(
                        DragValue::new(&mut self.view_cfg.viewed_eigenstate)
                            .clamp_range(0..=energies.len() - 1),
                    )
                    .changed();
                ui.label(format!(
                    "Energy: {}",
                    energies[self.view_cfg.viewed_eigenstate]
                ));
                needs_update |= ui
                    .checkbox(&mut self.view_cfg.show_probability, "Show probability")
                    .changed();
                needs_update |= ui
                    .checkbox(&mut self.view_cfg.show_force_field, "Show force field")
                    .changed();
            }

            ui.separator();
            ui.strong("Config");
            needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.sim.cfg.v0)
                        .speed(1e-1)
                        .prefix("V0: "),
                )
                .changed();

            needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.sim.cfg.v_soft)
                        .speed(1e-3)
                        .clamp_range(1e-5..=10.0)
                        .prefix("V softening: "),
                )
                .changed();

            ui.horizontal(|ui| {
                needs_recalculate |= ui
                    .selectable_value(
                        &mut self.sim.cfg.potental_mode,
                        PotentialMode::Delta,
                        "delta(r-r')",
                    )
                    .changed();
                needs_recalculate |= ui
                    .selectable_value(
                        &mut self.sim.cfg.potental_mode,
                        PotentialMode::Kqr,
                        "kq / (r + soft)",
                    )
                    .changed();
            });

            needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.sim.cfg.grid_width)
                        .speed(1e-1)
                        .prefix("Grid size: ")
                        .clamp_range(8..=1000),
                )
                .changed();

            needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.sim.cfg.tolerance)
                        .speed(1e-1)
                        .prefix("Tolerance: "),
                )
                .changed();

            needs_recalculate |= ui
                .add(
                    DragValue::new(&mut self.sim.cfg.num_solver_iters)
                        .speed(1e-1)
                        .prefix("Max iters: "),
                )
                .changed();

            ui.horizontal(|ui| {
                needs_recalculate |= ui
                    .add(
                        DragValue::new(&mut self.sim.cfg.n_states)
                            .speed(1e-1)
                            .prefix("Max states: "),
                    )
                    .changed();

                //ui.checkbox(&mut self.max_states_is_grid_width, "From width");
            });

            ui.separator();
            ui.strong("Nuclei");
            needs_recalculate |= nucleus_editor(ui, &mut self.sim.state.nuclei);

            ui.separator();
            ui.strong("Energy levels");
            needs_update |= electric_editor(
                ui,
                &mut self.view_cfg,
                &mut self.sim.state,
                self.sim.artefacts.as_ref(),
            );
        });

        CentralPanel::default().show(ctx, |ui| {
            if let Some(art) = &self.sim.artefacts {
                needs_recalculate |= self.img.show(
                    "Main image".into(),
                    ui,
                    &self.view_cfg,
                    &mut self.sim.state,
                    art,
                ).dragged();
            }
        });

        if needs_recalculate {
            self.recalculate(ctx);
        } else if needs_update {
            self.update_view(ctx);
        }
    }
}

impl TemplateApp {
    fn recalculate(&mut self, ctx: &egui::Context) {
        self.sim.recalculate();
        self.update_view(ctx);
    }

    fn update_view(&mut self, ctx: &egui::Context) {
        if let Some(artefacts) = self.sim.artefacts() {
            self.view_cfg.viewed_eigenstate = self
                .view_cfg
                .viewed_eigenstate
                .min(artefacts.energies.len() - 1);

            /*self.img.set_state(
                "Spronkus".into(),
                ctx,
                &self.view_cfg,
                &self.sim.state(),
                artefacts,
            );
            */
        }
    }
}

fn initial_state(cfg: &SimConfig) -> SimState {
    SimState {
        energy_level: 0,
        coeffs: (0..cfg.grid_width.pow(2))
            .map(|n| if n == 0 { 1.0 } else { 0.0 })
            .collect(),
        nuclei: vec![Nucleus {
            pos: Vec2::new(cfg.grid_width as f32 / 2., cfg.grid_width as f32 / 2.),
            vel: Vec2::ZERO,
        }],
    }
}

const N: usize = 20;
fn initial_cfg() -> SimConfig {
    SimConfig {
        potental_mode: PotentialMode::Delta,
        dx: 1.0,
        grid_width: N,
        v0: -1.,
        v_soft: 0.1,
        v_scale: 1.,
        n_states: 10,
        num_solver_iters: 30,
        tolerance: 1e-2,
    }
}
