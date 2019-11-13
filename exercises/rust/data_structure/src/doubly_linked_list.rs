// https://play.rust-lang.org/?gist=c3db81ec94bf231b721ef483f58deb35
// https://gist.github.com/anonymous/c3db81ec94bf231b721ef483f58deb35

use std::cell::RefCell;
use std::rc::{Rc, Weak};
use std::fmt::Display;


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

    fn append(node: &mut Rc<RefCell<Node<T>>>, data: T) -> Option<Rc<RefCell<Node<T>>>> {
        let old_next_link = node.borrow_mut().next.take();
        let new_node = Rc::new(RefCell::new(Node::new(data)));
        new_node.borrow_mut().prev = Some(Rc::downgrade(node));
        new_node.borrow_mut().next = old_next_link.clone();
        node.borrow_mut().next = Some(new_node.clone());
        old_next_link.map(|n| n.borrow_mut().prev = Some(Rc::downgrade(&new_node)));
        Some(new_node)
    }
}

impl<T> DoublyLinkedList<T> {
    pub fn new() -> Self {
        Self { first: None, last: None }
    }

    pub fn append(&mut self, data: T) {
        if self.last.is_none() {
            let rc = Rc::new(RefCell::new(Node::new(data)));
            self.first = Some(rc.clone());
            self.last = Some(rc);
        } else {
            let mut last_node = self.last.take().unwrap();
            self.last = Node::append(&mut last_node, data);
            self.first = self.last.clone();
        }
    }

}

impl<T: Display> Display for DoublyLinkedList<T> {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        write!(w, "[")?;
        let mut node = self.first.clone();
        while let Some(n) = node {
            write!(w, "{}", n.borrow().data)?;
            node = n.borrow().next.clone();
            if node.is_some() {
                write!(w, ", ")?;
            }
        }
        write!(w, "]")
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linked_list_append() {
        // cargo test -- --nocapture
        let mut l: DoublyLinkedList<u32> = DoublyLinkedList::new();
        l.append(1);
        l.append(2);
        l.append(3);
        println!("{}", l)
    }

}
