use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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

    pub fn find(&self, key: K) -> Option<&V> {
        let bn = calc_hash_bucket(&key, BUCKETS_SIZE);
        let mut current = &self.buckets[bn];
        while let Some(node) = current.as_ref() {
            if node.key == key {
                return Some(&node.elem)
            }
            current = &node.next;
        }
        None
    }

    pub fn remove(&mut self, key: K) -> Option<V> {
        // https://codereview.stackexchange.com/questions/169523/linked-list-with-removal-function-in-rust
        let bn = calc_hash_bucket(&key, BUCKETS_SIZE);
        let mut current = &mut self.buckets[bn];
        loop {
            match current {
                None => return None,
                Some(node) if node.key == key => {
                    let r = Some(node.elem.clone());
                    *current = node.next.take();
                    return r;
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
        assert_eq!(h.find(1), Some(&("123".to_string())));
        assert_eq!(h.find(3), Some(&("789".to_string())));
    }

    #[test]
    fn test_hashtable_insert_and_delete() {
        let mut h: HashTable<u32, String> = HashTable::new();
        h.insert(1, "123".to_string());
        h.insert(2, "456".to_string());
        h.insert(3, "789".to_string());
        assert_eq!(h.find(3), Some(&("789".to_string())));
        assert_eq!(h.remove(3), Some("789".to_string()));
        assert_eq!(h.remove(2), Some("456".to_string()));
        assert_eq!(h.find(3), None);
        assert_eq!(h.find(2), None);
        assert_eq!(h.find(1), Some(&("123".to_string())));
    }

}
