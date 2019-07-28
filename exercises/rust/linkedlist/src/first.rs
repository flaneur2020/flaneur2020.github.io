use std::mem;


struct Node {
    elem: i32,
    next: Option<Box<Node>>,
}

pub struct LinkedStack {
    head: Option<Box<Node>>,
}


impl LinkedStack {
    pub fn new() -> LinkedStack {
        LinkedStack { head: None }
    }

    pub fn push(&mut self, v: i32) {
        let n = Node {
            elem: v,
            next: mem::replace(&mut self.head, None),
        };
        self.head = Some(Box::new(n));
    }

    pub fn pop(&mut self) ->  i32 {
        0
    }
}


#[test]
fn test_list() {
    // cargo test -- --nocapture
}
