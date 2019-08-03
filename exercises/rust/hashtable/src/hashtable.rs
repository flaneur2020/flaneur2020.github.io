use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const BUCKETS_SIZE: usize = 32;

pub struct HashTable<K: Hash+Eq+Clone, V: Clone> {
    buckets: Vec<Link<K, V>>,
}

#[derive(Clone)]
struct Node<K: Hash+Eq+Clone, V: Clone> {
    key: K,
    elem: V,
    next: Link<K, V>,
}

type Link<K: Hash, V> = Option<Box<Node<K, V>>>;

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
        let bn = calc_hash_bucket(&key, BUCKETS_SIZE);
        self.buckets[bn] = match self.buckets[bn].take() {
            None => {
                Some(Box::new(Node {
                    key: key,
                    elem: elem,
                    next: None,
                }))
            }
            Some(boxed_node) => {
                Some(Box::new(Node {
                    key: key,
                    elem: elem,
                    next: Some(boxed_node)
                }))
            }
        }
    }

    pub fn get(&self, key: K) -> Option<&V> {
        let bn = calc_hash_bucket(&key, BUCKETS_SIZE);
        let mut current = &self.buckets[bn];
        while let None = current {
            let node = current.as_ref().unwrap();
            if node.key == key {
                return Some(&node.elem)
            }
            current = &node.next;
        }
        None
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
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
