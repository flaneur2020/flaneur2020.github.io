use log;
use proxy_wasm::traits::{HttpContext, Context};
use proxy_wasm::types::{Action, LogLevel};

#[no_mangle]
pub fn _start() {
    proxy_wasm::set_log_level(LogLevel::Trace);
    proxy_wasm::set_http_context(|context_id, _| -> Box<dyn HttpContext> {
        Box::new(SecretTokenFilter { context_id })
    });
}


struct SecretTokenFilter {
    context_id: u32,
}

impl Context for SecretTokenFilter {}

impl HttpContext for SecretTokenFilter {
    fn on_http_request_headers(&mut self, _: usize) -> Action {
        log::info!("===== first!");

        match self.get_http_request_header("X-TOKEN") {
            Some(token) if token == "secret123" => {
                self.resume_http_request();
                Action::Continue
            },
            _ => {
                self.send_http_response(403, vec![("X-SECRET-FILTER", "0.1.0")], Some(b"Access Forbidden..."));
                Action::Pause
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
