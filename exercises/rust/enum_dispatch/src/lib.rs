use enum_dispatch::enum_dispatch;

#[enum_dispatch(BirdEnum)]
trait Bird {
    fn fly(&self);
}

#[enum_dispatch]
enum BirdEnum {
    Duck,
    Swift,
}

struct Duck;

impl Bird for Duck {
    fn fly(&self) {
        println!("Duck flying");
    }
}

struct Swift;

impl Bird for Swift {
    fn fly(&self) {
        println!("Swift flying");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut birds: Vec<BirdEnum> = vec![];
        birds.push(Duck{}.into());
        birds.push(Swift{}.into());
        for bird in birds {
            bird.fly();
        }
    }
}
