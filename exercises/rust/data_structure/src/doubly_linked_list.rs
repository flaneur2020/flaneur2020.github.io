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
    last: Option<Weak<RefCell<Node<T>>>>,
}

impl<T> Node<T> {
    fn new(data: T) -> Self {
        Self {
            data: data,
            next: None,
            prev: None,
        }
    }

    fn append(node: &mut Rc<RefCell<Node<T>>>, data: T) -> Option<Rc<RefCell<Node<T>>>> {
        let mut new_node = Node::new(data);
        let is_last = node.borrow().next.is_none();
        if is_last {
            new_node.prev = Some(Rc::downgrade(node));
            let new_rc = Rc::new(RefCell::new(new_node));
            node.borrow_mut().next = Some(new_rc.clone());
            Some(new_rc)
        } else {
            let old_next_link = node.borrow_mut().next.take();
            new_node.prev = Some(Rc::downgrade(node));
            new_node.next = old_next_link.clone();
            let new_rc = Rc::new(RefCell::new(new_node));
            node.borrow_mut().next = Some(new_rc.clone());
            old_next_link.unwrap().borrow_mut().prev = Some(Rc::downgrade(&new_rc));
            Some(new_rc)
        }
    }
}

impl<T> DoublyLinkedList<T> {
    pub fn new() -> Self {
        Self { first: None, last: None }
    }
}
