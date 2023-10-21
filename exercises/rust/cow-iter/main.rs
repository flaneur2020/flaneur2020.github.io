use std::{borrow::Cow};

struct VecIterMut<'a> {
    v: &'a mut Vec<f32>,
    pos: usize,
}

impl<'a> Iterator for VecIterMut<'a> {
    type Item = &'a mut f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.v.len() {
            return None;
        }

        let r = Some(self.v.get_mut(self.pos)?);
        self.pos += 1;
        r
    }
}

fn main() {}