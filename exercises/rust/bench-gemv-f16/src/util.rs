use std::sync::Once;

static INIT_RAYON: Once = Once::new();

pub fn init_rayon() {
    INIT_RAYON.call_once(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build_global()
            .unwrap();
    });
}
