#[derive(Default)]
enum SuccessKind {
    #[default]
    One,
}

#[derive(Default)]
enum Status {
    Pending,        // Valid - this is a unit variant
    #[default]
    Success(SuccessKind) // Cannot be default - requires data
}

fn main() {
    println!("Hello, world!");
}
