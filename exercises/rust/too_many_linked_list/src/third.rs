use std::mem;
use std::rc::Rc;

struct Node<T> {
    elem: T,
    next: Link<T>,
}

type Link<T> = Option<Rc<Node<T>>>;

pub struct LinkedList<T> {
    head: Link<T>,
}

pub struct IntoIter<T>(LinkedList<T>);

pub struct Iter<'a, T> {
    next: Option<&'a Node<T>>,
}

pub struct IterMut<'a, T> {
    next: Option<&'a mut Node<T>>,
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        LinkedList { head: None }
    }

    pub fn push(&mut self, v: T) {
        let n = Node {
            elem: v,
            next: self.head.clone(),
        };
        self.head = Some(Rc::new(n));
    }

    pub fn peek(&mut self) -> Option<&T> {
        // Option<T>#as_ref() 将 Option<T> 转为 Option<&T>
        self.head.as_ref().map(|n| {
            &n.elem
        })
    }

    pub fn tail(&self) -> LinkedList<T> {
        LinkedList {
            head: self.head.as_ref().and_then(|n| n.next.clone())
        }
    }

    pub fn into_iter(self) -> IntoIter<T> {
        // into_iter() 是按 move 语义遍历集合对象
        IntoIter(self)
    }

    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        Iter {
            next: self.head.as_ref().map(|n| &**n),
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next {
            None => None,
            Some(node) => {
                self.next = node.next.as_ref().map(|n| &**n);
                Some(&node.elem)
            }
        }
    }
}

impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        let mut current = self.head.take();
        while let Some(rc_node) = current {
            // Rc::try_unwrap： 当 rc 引用计数为零时返回 Ok
            if let Ok(mut node) = Rc::try_unwrap(rc_node) {
                current = node.next.take();
            } else {
                break
            }
        }
    }
}


#[test]
fn test_list() {
    let mut l: LinkedList<i32> = LinkedList::new();
    l.push(1);
    l.push(2);
    assert_eq!(l.peek(), Some(&2));
}
