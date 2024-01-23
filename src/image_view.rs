use egui::{
    self,
    epaint::{ColorImage, ImageData, ImageDelta, TextureId},
    vec2, Image, Sense, TextureOptions, Ui, Vec2,
};
use ndarray::{Array3, Array2};
use qdynamics::sim::{SimState, SimArtefacts};

pub struct StateViewConfig {
    pub energy_level: usize,
}

#[derive(Default)]
pub struct ImageViewWidget {
    tex: Option<TextureId>,
}

impl ImageViewWidget {
    const OPTS: TextureOptions = TextureOptions::NEAREST;

    pub fn show(&mut self, ui: &mut Ui) -> egui::Response {
        if let Some(tex) = self.tex {
            let available = ui.available_size();
            if let Some(tex_meta) = ui.ctx().tex_manager().read().meta(tex) {
                let tex_size = Vec2::from(tex_meta.size.map(|v| v as f32));
                let tex_aspect = tex_size.x / tex_size.y;
                let size = if available.x / available.y < tex_aspect {
                    vec2(available.x, available.x / tex_aspect)
                } else {
                    vec2(available.y / tex_aspect, available.y)
                };

                let image = Image::new((tex, size);
                return ui.add().sense(Sense::click_and_drag()));
            }
        }

        ui.label("Texture not set, this is an error!")
    }

    pub fn set_state(&mut self, name: String, ctx: &egui::Context, state: &SimState, artefact: &SimArtefacts) {
        let image = array_to_imagedata(artefact.eigenstates[]);

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

fn sim_state_editor(ui: &mut Ui, state: &mut SimState) -> bool {
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
