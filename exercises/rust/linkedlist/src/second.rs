use std::mem;

struct Node<T> {
    elem: T,
    next: Option<Box<Node<T>>>,
}

pub struct LinkedStack<T> {
    head: Option<Box<Node<T>>>,
}

pub struct IntoIter<T>(LinkedStack<T>);

pub struct Iter<'a, T> {
    next: Option<&'a Node<T>>,
}

pub struct IterMut<'a, T> {
    next: Option<&'a mut Node<T>>,
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
        // Option<T>#as_ref() 将 Option<T> 转为 Option<&T>
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

    pub fn into_iter(self) -> IntoIter<T> {
        // into_iter() 是按 move 语义遍历集合对象
        IntoIter(self)
    }

    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        Iter {
            next: self.head.as_ref().map(|n| &**n),
        }
    }

    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, T> {
        IterMut {
            next: self.head.as_mut().map(|n| &mut **n)
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
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

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        // take() 的作用类似 mem::replace，将 Option<T> 中的值对换出来，转移 ownership
        match self.next.take() {
            None => None,
            Some(node) => {
                self.next = node.next.as_mut().map(|n| &mut **n);
                Some(&mut node.elem)
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

#[test]
fn test_list_into_iter() {
    let mut l: LinkedStack<i32> = LinkedStack::new();
    l.push(1);
    l.push(2);
    l.push(3);
    let mut iter = l.into_iter();
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_list_iter() {
    let mut l: LinkedStack<i32> = LinkedStack::new();
    l.push(1);
    l.push(2);
    l.push(3);
    let mut iter = l.iter();
    assert_eq!(iter.next(), Some(&3));
    assert_eq!(iter.next(), Some(&2));
    assert_eq!(iter.next(), Some(&1));
    assert_eq!(iter.next(), None);
    assert_eq!(l.peek(), Some(&3));
}
