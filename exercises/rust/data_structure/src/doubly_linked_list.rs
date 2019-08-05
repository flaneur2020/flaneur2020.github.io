// https://play.rust-lang.org/?gist=c3db81ec94bf231b721ef483f58deb35
// https://gist.github.com/anonymous/c3db81ec94bf231b721ef483f58deb35

use std::cell::RefCell;
use std::rc::{Rc, Weak};

struct Node<T> {
    data: T,
    next: Option<Rc<RefCell<Node<T>>>>,
    prev: Option<Weak<RefCell<Node<T>>>>,
}

struct DoublyLinkedList<T> {
    first: Option<Rc<RefCell<Node<T>>>>,
    last: Option<Rc<RefCell<Node<T>>>>,
}

impl<T> Node<T> {
    fn new(data: T) -> Self {
        Self {
            data: data,
            next: None,
            prev: None,
        }
    }

    fn append(node: &mut Rc<RefCell<Node<T>>>, data: T) -> {
    }
}
