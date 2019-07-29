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

    pub fn peek(&mut self) -> Option<i32> {
        match &self.head {
            None => None,
            Some(n) => Some((*n).elem),
        }
    }

    pub fn pop(&mut self) -> Option<i32> {
        match mem::replace(&mut self.head, None) {
            None => None,
            Some(boxed_node) => {
                self.head = boxed_node.next;
                Some(boxed_node.elem)
            }
        }
    }
}

impl Drop for LinkedStack {
    fn drop(&mut self) {
        // let mut current = mem::replace(&mut self.head, None);
        // while let Some(mut boxed_node) = current {
        //    current = mem::replace(&mut boxed_node.next, None);
        // }
        let mut current = self.head.take();
        while let Some(mut boxed_node) = current {
            current = boxed_node.next.take();
        }
    }
}

// Note:
//
// can not move out of borrowed content 是什么意思？
// &mut self 是一个引用，如果 move 引用中的字段，便会报这个错。
//
// mem::replace 似乎可以将引用对象中的字段 move 给另一个变量；
// 遇到 cannot move out of borrowed content 可以使用它来处理？
// https://github.com/rust-unofficial/patterns/blob/master/idioms/mem-replace.md
//
// 为什么需要写 Drop？


#[test]
fn test_list() {
    let mut l = LinkedStack::new();
    l.push(1);
    l.push(2);
    assert_eq!(l.peek(), Some(2));
    assert_eq!(l.pop(), Some(2));
    assert_eq!(l.pop(), Some(1));
}
