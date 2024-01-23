use egui::{
    self,
    epaint::{ColorImage, ImageData, ImageDelta, TextureId},
    vec2, Image, Sense, Stroke, TextureOptions, Ui, Vec2,
};
use egui::{
    CentralPanel, Color32, DragValue, Frame, Pos2, Rect, Response, SelectableLabel, SidePanel,
};
use ndarray::{Array2, Array3};
use qdynamics::sim::{Nucleus, SimArtefacts, SimState};

#[derive(Clone, Copy)]
pub struct StateViewConfig {
    pub viewed_eigenstate: usize,
    pub show_probability: bool,
}

impl Default for StateViewConfig {
    fn default() -> Self {
        Self {
            viewed_eigenstate: 0,
            show_probability: false,
        }
    }
}

#[derive(Default)]
pub struct ImageViewWidget {
    tex: Option<TextureId>,
    state: Option<SimState>,
}

impl ImageViewWidget {
    const OPTS: TextureOptions = TextureOptions::NEAREST;

    pub fn show(&mut self, ui: &mut Ui) -> egui::Response {
        if let Some(tex) = self.tex {
            let available = ui.available_size();
            if let Some(tex_meta) = ui.ctx().tex_manager().read().meta(tex) {
                let tex_size = Vec2::from(tex_meta.size.map(|v| v as f32));
                let tex_aspect = tex_size.x / tex_size.y;
                let image_size_egui = if available.x / available.y < tex_aspect {
                    vec2(available.x, available.x / tex_aspect)
                } else {
                    vec2(available.y / tex_aspect, available.y)
                };

                Frame::canvas(ui.style()).show(ui, |ui| {
                    let resp = ui.allocate_response(image_size_egui, Sense::click_and_drag());

                    let paint = ui.painter();
                    paint.image(
                        tex,
                        resp.rect,
                        Rect::from_two_pos(Pos2::ZERO, Pos2::new(1., 1.)),
                        Color32::WHITE,
                    );

                    if let Some(state) = &self.state {
                        let egui_coord_per_sim_coord = |pt: egui::Vec2| {
                            resp.rect.min
                                + ((pt + egui::Vec2::splat(0.5)) * image_size_egui)
                                    / egui::Vec2::from(tex_meta.size.map(|sz| sz as f32))
                        };

                        for nucleus in &state.nuclei {
                            paint.circle(
                                egui_coord_per_sim_coord(egui::Vec2::from(nucleus.pos.to_array())),
                                7.0,
                                Color32::GREEN,
                                Stroke::NONE,
                            );
                        }
                    }

                    return resp;
                });
            }
        }

        ui.label("Texture not set, this is an error!")
    }

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

        if let Some(tex) = self.tex {
            ctx.tex_manager()
                .write()
                .set(tex, ImageDelta::full(image, Self::OPTS))
        } else {
            self.tex = Some(ctx.tex_manager().write().alloc(name, image, Self::OPTS))
        }
    }

    pub fn tex(&self) -> Option<TextureId> {
        self.tex
    }
}

pub fn sim_state_editor(ui: &mut Ui, state: &mut SimState) -> bool {
    let mut needs_recalculate = false;

    // Just nuclei for now
    let mut delete = None;
    for (idx, nucleus) in state.nuclei.iter_mut().enumerate() {
        ui.horizontal(|ui| {
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

pub fn display_imagedata(cfg: &StateViewConfig, artefacts: &SimArtefacts) -> ImageData {
    let eigstate = &artefacts.eigenstates[cfg.viewed_eigenstate];

    let image;
    if cfg.show_probability {
        let sum: f32 = eigstate.iter().map(|v| v.powi(2)).sum();
        image = eigstate.map(|v| {
            let v = (v.powi(2) / sum) as f32;
            let v = 100.0 * v;
            [v, v, v, 0.0]
        });
    } else {
        image = eigstate.map(|v| {
            let v = *v as f32 * (artefacts.potential.ncols() as f32).sqrt();
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