//extern crate wasm_bindgen;
#![feature(new_uninit)]

use wasm_bindgen::prelude::*;
use algorithm::{FullPolytope, ReverseSearchOut, reverse_search};
use std::rc::Rc;
use anyhow::{anyhow, Result};
use std::convert::TryFrom;
use web_sys::console;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello,{}!", name));
}

// This is like the `main` function, except for JavaScript.
#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    // This provides better error messages in debug mode.
    // It's disabled in release mode so it doesn't bloat up the file size.
    #[cfg(debug_assertions)]
    console_error_panic_hook::set_once();
    console::log_1(&JsValue::from_str("Hello from rust!"));
    Ok(())
}

fn js_func_wrapper<'a>(callback: &'a js_sys::Function) -> impl FnMut(ReverseSearchOut) -> Result<()> + 'a{
    move |rs_out: ReverseSearchOut| {
        console::log_1(&JsValue::from_str("Writer callback"));
        let this = JsValue::null();
        let param_vec = js_sys::Float64Array::from(&rs_out.param.to_vec()[..]);
        let js_length = js_sys::Number::from(u32::try_from(rs_out.minkowski_decomp.len())?);
        let minkowski_decomp = js_sys::Uint8Array::new(&js_length);
        for i in 0..rs_out.minkowski_decomp.len(){
            let downsized = u8::try_from(rs_out.minkowski_decomp[i])?;
            minkowski_decomp.set_index(u32::try_from(i)?, downsized);
        }
        let result = callback.call2(&this, &param_vec, &minkowski_decomp);
        result.map_err(|_| anyhow!("Error calling writer callback"))?;
        Ok(())
    }
}

#[wasm_bindgen(js_name = reverseSearch)]
pub fn reverse_search_wrapper(polytope_data: &[f32], 
                              num_polytopes: usize,
                              num_vertices: usize,
                              dim: usize,
                              writer_callback: &js_sys::Function) -> std::result::Result<(), String>{
    console::log_1(&JsValue::from_str("Entering reverse search wrapper"));
    let mut poly_list = Vec::<FullPolytope>::with_capacity(num_polytopes);
    console::log_1(&JsValue::from_str("Copying data"));
    for i in 0..num_polytopes{
        let mut polytope_matrix = Vec::<Rc<[f64]>>::with_capacity(num_vertices);
        for j in 0..num_vertices{
            let mut uninit_vector = Rc::<[f64]>::new_uninit_slice(dim);
            let data = Rc::get_mut(&mut uninit_vector).unwrap();
            for k in 0..dim{
                let js_index = i * num_vertices * dim + j * dim + k;
                data[k].write(polytope_data[js_index].into());
            }
            let vector: Rc<[f64]> = unsafe { uninit_vector.assume_init() };
            polytope_matrix.push(vector);
        }
        let polytope = FullPolytope::new(polytope_matrix);
        poly_list.push(polytope);
    }
    let wrapped_writer_callback = js_func_wrapper(writer_callback);
    console::log_1(&JsValue::from_str("Starting search"));
    let rs_result = reverse_search(&mut poly_list, Box::new(wrapped_writer_callback));
    match rs_result{
        Ok(_) => Ok(()),
        Err(err) => Err(format!("{:#?}", err).to_string())
    }

}