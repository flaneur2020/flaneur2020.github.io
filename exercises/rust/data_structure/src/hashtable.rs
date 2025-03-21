use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem;

const BUCKETS_SIZE: usize = 32;

#[derive(Debug)]
pub struct HashTable<K: Hash+Eq+Clone, V: Clone> {
    buckets: Vec<Link<K, V>>,
}

#[derive(Clone,Debug)]
struct Node<K: Hash+Eq+Clone, V: Clone> {
    key: K,
    elem: V,
    next: Link<K, V>,
}

type Link<K: Hash, V> = Option<Box<Node<K, V>>>;

impl<K, V> Node<K, V>
    where K: Hash+Eq+Clone,
          V: Clone
{
    pub fn take_elem(self) -> V {
        self.elem
    }
}

impl<K, V> HashTable<K, V>
    where K: Hash+Eq+Clone,
          V: Clone
{

    pub fn new() -> Self {
        let mut buckets: Vec<Link<K, V>> = vec![None; BUCKETS_SIZE];
        HashTable {
            buckets: buckets,
        }
    }

    pub fn insert(&mut self, key: K, elem: V) {
        match self.find_mut(&key) {
            None => self._prepend(key, elem),
            Some(m) => {
                mem::replace(m, elem);
            }
        }
    }

    fn _prepend(&mut self, key: K, elem: V) {
        let bn = calc_hash_bucket(&key, BUCKETS_SIZE);
        let head = self.buckets[bn].take();
        self.buckets[bn] = Some(Box::new(Node {
            key: key,
            elem: elem,
            next: head,
        }))
    }

    pub fn find(&self, key: &K) -> Option<&V> {
        let bn = calc_hash_bucket(key, BUCKETS_SIZE);
        let mut current = &self.buckets[bn];
        while let Some(node) = current.as_ref() {
            if node.key == *key {
                return Some(&node.elem)
            }
            current = &node.next;
        }
        None
    }

    pub fn find_mut(&mut self, key: &K) -> Option<&mut V> {
        let bn = calc_hash_bucket(key, BUCKETS_SIZE);
        let mut current = &mut self.buckets[bn];
        while let Some(node) = current.as_mut() {
            if node.key == *key {
                return Some(&mut node.elem)
            }
            current = &mut node.next;
        }
        None
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        // https://codereview.stackexchange.com/questions/169523/linked-list-with-removal-function-in-rust
        let bn = calc_hash_bucket(key, BUCKETS_SIZE);
        let mut current = &mut self.buckets[bn];
        loop {
            match current {
                None => return None,
                Some(node) if node.key == *key => {
                    // *current 复制移动走了之后，node 的生命周期也结束了，不能再访问 node
                    // *current = node.next.take();
                    let next_node = node.next.take();
                    let old_node = mem::replace(current, next_node).unwrap();
                    return Some(old_node.take_elem());
                },
                Some(node) => {
                    current = &mut node.next
                }
            }
        }
    }
}


impl<K, V> Drop for HashTable<K, V>
    where K: Hash+Eq+Clone,
          V: Clone
{
    fn drop(&mut self) {
        for bucket in self.buckets.iter_mut() {
            let mut current = bucket.take();
            while let Some(mut boxed_node) = current {
                current = boxed_node.next.take();
            }
        }
    }
}


fn calc_hash_bucket<T: Hash>(t: &T, nbuckets: usize) -> usize {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    let h = s.finish();
    (h as usize) % nbuckets
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hashtable_insert_and_get() {
        let mut h: HashTable<u32, String> = HashTable::new();
        h.insert(1, "123".to_string());
        h.insert(2, "456".to_string());
        h.insert(3, "789".to_string());
        assert_eq!(h.find(&1), Some(&("123".to_string())));
        assert_eq!(h.find(&3), Some(&("789".to_string())));
    }

    #[test]
    fn test_hashtable_insert_and_delete() {
        let mut h: HashTable<u32, String> = HashTable::new();
        h.insert(1, "123".to_string());
        h.insert(2, "456".to_string());
        h.insert(3, "789".to_string());
        h.insert(3, "789789".to_string());
        assert_eq!(h.find(&3), Some(&("789789".to_string())));
        assert_eq!(h.remove(&3), Some("789789".to_string()));
        assert_eq!(h.remove(&2), Some("456".to_string()));
        assert_eq!(h.find(&3), None);
        assert_eq!(h.find(&2), None);
        assert_eq!(h.find(&1), Some(&("123".to_string())));
    }

}
