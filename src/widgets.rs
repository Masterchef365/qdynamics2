use egui::{
    self,
    epaint::{ColorImage, ImageData, ImageDelta, TextureId},
    vec2, Image, ScrollArea, Sense, Stroke, TextureOptions, Ui, Vec2,
};
use egui::{
    CentralPanel, Color32, DragValue, Frame, Pos2, Rect, Response, SelectableLabel, SidePanel,
};
use ndarray::{Array2, Array3};
use qdynamics::sim::{
    calculate_classical_force, calculate_electric_force, compute_force_at, Nucleus, SimConfig,
    SimElectronicState, SimState,
};

#[derive(Clone, Copy)]
pub struct StateViewConfig {
    pub show_probability: bool,
    pub show_force_field: bool,
}

impl Default for StateViewConfig {
    fn default() -> Self {
        Self {
            show_probability: true,
            show_force_field: false,
        }
    }
}

#[derive(Default)]
pub struct ImageViewWidget {
    tex: Option<TextureId>,
}

impl ImageViewWidget {
    const OPTS: TextureOptions = TextureOptions::LINEAR;

    pub fn show(
        &mut self,
        name: String,
        ui: &mut Ui,
        view: &StateViewConfig,
        state: &mut SimState,
        art: &SimElectronicState,
        cfg: &SimConfig,
    ) -> egui::Response {
        let image = display_imagedata(view, state, art);

        let ctx = ui.ctx();
        if let Some(tex) = self.tex {
            ctx.tex_manager()
                .write()
                .set(tex, ImageDelta::full(image, Self::OPTS))
        } else {
            self.tex = Some(ctx.tex_manager().write().alloc(name, image, Self::OPTS))
        }

        if let Some(tex) = self.tex {
            let available = ui.available_size();
            let tex_meta = ui.ctx().tex_manager();
            let tex_meta = tex_meta.read();
            let tex_meta = tex_meta.meta(tex).unwrap();

            let tex_size = Vec2::from(tex_meta.size.map(|v| v as f32));
            let tex_aspect = tex_size.x / tex_size.y;
            let image_size_egui = if available.x / available.y < tex_aspect {
                vec2(available.x, available.x / tex_aspect)
            } else {
                vec2(available.y / tex_aspect, available.y)
            };

            Frame::canvas(ui.style())
                .show(ui, |ui| {
                    let resp = ui.allocate_response(image_size_egui, Sense::click_and_drag());

                    // Draw background image
                    let paint = ui.painter(); // TODO: use clipping painter_at(). This is nice for
                                              // debugging as is though...
                    paint.image(
                        tex,
                        resp.rect,
                        Rect::from_two_pos(Pos2::ZERO, Pos2::new(1., 1.)),
                        Color32::WHITE,
                    );

                    let tex_size = egui::Vec2::from(tex_meta.size.map(|sz| sz as f32));

                    let sim_coord_to_egui_coord = |sim_coord: glam::Vec2| {
                        let pt = egui::Vec2::from(sim_coord.to_array());
                        resp.rect.min + (pt + egui::Vec2::splat(0.5)) * image_size_egui / tex_size
                        //+ pt * image_size_egui / tex_size
                    };

                    /*
                    let egui_coord_to_sim_coord = |egui_coord: egui::Pos2| {
                        (egui_coord - resp.rect.min) * tex_size / image_size_egui
                    };
                    */

                    let egui_vect_to_sim_vect = |egui_vect: egui::Vec2| {
                        let v = egui_vect * tex_size / image_size_egui;
                        glam::Vec2::new(v.x, v.y)
                    };

                    // Move nuclei
                    if resp.dragged() {
                        if let Some(pointer_pos) = resp.interact_pointer_pos() {
                            for nucleus in &mut state.nuclei {
                                let screen_pos = sim_coord_to_egui_coord(nucleus.pos);
                                let screen_dist = screen_pos - pointer_pos;

                                let drag_radius = 20.0;
                                if screen_dist.length() < drag_radius {
                                    nucleus.pos += egui_vect_to_sim_vect(resp.drag_delta());
                                    break;
                                }
                            }
                        }
                    }

                    let psi = &art.eigenstates[state.energy_level];

                    let display_mult = 50.;

                    // Draw nuclei
                    for (idx, nucleus) in state.nuclei.iter().enumerate() {
                        let center = sim_coord_to_egui_coord(nucleus.pos);
                        paint.circle(center, 7.0, Color32::GREEN, Stroke::NONE);

                        // Velocity arrow
                        //paint.arrow(center, egui::Vec2::from(nucleus.vel.to_array()), Stroke::new(1.0, Color32::RED));

                        // Acceleration arrow
                        let electric_force =
                            calculate_electric_force(&art, state.energy_level, nucleus.pos);
                        let nuclear_force = calculate_classical_force(idx, state, cfg);

                        let total_force = electric_force + nuclear_force;
                        paint.arrow(
                            center,
                            egui::Vec2::from(electric_force.to_array()) * display_mult,
                            Stroke::new(2.0, Color32::RED),
                        );
                        paint.arrow(
                            center,
                            egui::Vec2::from(nuclear_force.to_array()) * display_mult,
                            Stroke::new(2.0, Color32::BLUE),
                        );
                        paint.arrow(
                            center,
                            egui::Vec2::from(total_force.to_array()) * display_mult,
                            Stroke::new(1.0, Color32::GREEN),
                        );
                    }

                    // Draw arrows for direction
                    if view.show_force_field {
                        for y in 0..psi.nrows() {
                            for x in 0..psi.ncols() {
                                let force = compute_force_at(art, state.energy_level, x, y);
                                //let force = force.normalize_or_zero();
                                paint.arrow(
                                    sim_coord_to_egui_coord(glam::Vec2::new(x as f32, y as f32)),
                                    egui::Vec2::new(force.x, force.y).normalized() * (force.length() * display_mult).min(20.),
                                    Stroke::new(2.0, Color32::DARK_GREEN),
                                );
                            }
                        }
                    }

                    resp
                })
                .inner
        } else {
            ui.label("Texture not set, this is an error!")
        }
    }

    /*
    pub fn set_state(
        &mut self,
        name: String,
        ctx: &egui::Context,
        cfg: &StateViewConfig,
        state: &SimState,
        artefact: &SimArtefacts,
    ) {
        let image = display_imagedata(cfg, artefact);
        self.state = Some(state.clone());
        self.artefact = Some(artefact.clone());

        if let Some(tex) = self.tex {
            ctx.tex_manager()
                .write()
                .set(tex, ImageDelta::full(image, Self::OPTS))
        } else {
            self.tex = Some(ctx.tex_manager().write().alloc(name, image, Self::OPTS))
        }
    }
    */

    pub fn tex(&self) -> Option<TextureId> {
        self.tex
    }
}

pub fn nucleus_editor(ui: &mut Ui, nuclei: &mut Vec<Nucleus>) -> bool {
    let mut needs_recalculate = false;

    // Just nuclei for now
    let mut delete = None;

    ScrollArea::vertical()
        .id_source("Nuclei")
        .max_height(200.)
        .show(ui, |ui| {
            for (idx, nucleus) in nuclei.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{:>3} ", idx));

                    needs_recalculate |= ui
                        .add(DragValue::new(&mut nucleus.pos.x).prefix("x: ").speed(1e-1))
                        .changed();
                    needs_recalculate |= ui
                        .add(DragValue::new(&mut nucleus.pos.y).prefix("y: ").speed(1e-1))
                        .changed();

                    if ui.button("Delete").clicked() {
                        delete = Some(idx);
                        needs_recalculate = true;
                    }

                    if ui.button("Stop").clicked() {
                        nucleus.vel = glam::Vec2::ZERO;
                        needs_recalculate = true;
                    }
                });
            }
        });

    if let Some(idx) = delete {
        nuclei.remove(idx);
        needs_recalculate = true;
    }

    if ui.button("Add").clicked() {
        nuclei.push(Nucleus::default());
        needs_recalculate = true;
    }

    needs_recalculate
}

pub fn electric_editor(
    ui: &mut Ui,
    view: &mut StateViewConfig,
    state: &mut SimState,
    artefacts: Option<&SimElectronicState>,
) -> bool {
    let mut needs_update = false;

    ScrollArea::vertical()
        .id_source("Electrics")
        .max_height(200.)
        .show(ui, |ui| {
            for (idx, coeff) in state.coeffs.iter_mut().enumerate() {
                if let Some(art) = artefacts {
                    if idx >= art.energies.len() {
                        break;
                    }
                }

                ui.horizontal(|ui| {
                    ui.label(format!("{:>3} ", idx));

                    needs_update |= ui.add(DragValue::new(coeff).prefix("Coeff: ")).changed();

                    needs_update |= ui
                        .radio_value(&mut state.energy_level, idx, "View")
                        .changed();

                    if let Some(art) = artefacts {
                        if let Some(energy) = art.energies.get(idx) {
                            ui.label(format!("Energy: {}", energy));
                        }
                    }
                });
            }
        });

    needs_update
}

pub fn display_imagedata(view: &StateViewConfig, state: &SimState, artefacts: &SimElectronicState) -> ImageData {
    let eigstate = &artefacts.eigenstates[state.energy_level];

    let image;
    if view.show_probability {
        let sum: f32 = eigstate.iter().map(|v| v.powi(2)).sum();
        image = eigstate.map(|v| {
            let v = (v.powi(2) / sum) as f32;
            let v = 100.0 * v;
            [v, v, v, 0.0]
        });
    } else {
        image = eigstate.map(|v| {
            let v = *v as f32 * (artefacts.ham.potential.ncols() as f32).sqrt();
            if v > 0. {
                [v, 0.3 * v, 0.0, 0.0]
            } else {
                [0.0, 0.3 * -v, -v, 0.0]
            }
        });
    }
    array_to_imagedata(&image)
}

/// Converts an image of 0 - 1 flaots into egui image data
pub fn array_to_imagedata(array: &Array2<[f32; 4]>) -> ImageData {
    let dims = [array.ncols(), array.nrows()];

    let array = array.t();

    let mut rgba: Vec<u8> = array
        .iter()
        .copied()
        .flatten()
        .map(|value| (value.clamp(0., 1.) * 255.0) as u8)
        .collect();

    // Set alpha to one. TODO: UNDO THIS!!
    rgba.iter_mut()
        .skip(3)
        .step_by(4)
        .for_each(|v| *v = u8::MAX);

    ImageData::Color(ColorImage::from_rgba_unmultiplied(dims, &rgba).into())
}
