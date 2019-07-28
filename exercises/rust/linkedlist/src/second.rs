use std::mem;


struct Node<T> {
    elem: T,
    next: Option<Box<Node<T>>>,
}

pub struct LinkedStack<T> {
    head: Option<Box<Node<T>>>,
}


impl<T> LinkedStack<T> {
    pub fn new() -> Self {
        LinkedStack { head: None }
    }

    pub fn push(&mut self, v: T) {
        let n = Node {
            elem: v,
            next: mem::replace(&mut self.head, None),
        };
        self.head = Some(Box::new(n));
    }

    pub fn peek(&mut self) -> Option<&T> {
        self.head.as_ref().map(|n| {
            &n.elem
        })
    }

    pub fn pop(&mut self) -> Option<T> {
        match mem::replace(&mut self.head, None) {
            None => None,
            Some(boxed_node) => {
                self.head = boxed_node.next;
                Some(boxed_node.elem)
            }
        }
    }
}

impl<T> Drop for LinkedStack<T> {
    fn drop(&mut self) {
        let mut current = mem::replace(&mut self.head, None);
        while let Some(mut boxed_node) = current {
            current = mem::replace(&mut boxed_node.next, None);
        }
    }
}


#[test]
fn test_list() {
    let mut l: LinkedStack<i32> = LinkedStack::new();
    l.push(1);
    l.push(2);
    assert_eq!(l.peek(), Some(&2));
    assert_eq!(l.pop(), Some(2));
    assert_eq!(l.pop(), Some(1));
}
