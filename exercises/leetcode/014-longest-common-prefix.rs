pub fn find_idx(strs: &Vec<String>) -> i32 {
    let min_length = strs.iter().map(|s| s.len()).min().unwrap();
    let mut i: usize = 0;
    while i < min_length {
        for j in 0..strs.len() {
            if strs[j].as_bytes()[i] != strs[0].as_bytes()[i] {
                return i as i32;
            }
        }
        i += 1;
    }
    i as i32
}

pub fn longest_common_prefix(strs: Vec<String>) -> String {
    if strs.len() == 0 {
        return String::from("");
    }
    if strs.len() == 1 {
        return strs[0].clone();
    }
    let idx = find_idx(&strs);
    if idx < 0 {
        return String::from("");
    }
    return String::from_utf8_lossy(&strs[0].as_bytes()[0..idx as usize]).into_owned();
}

fn main() {
    let strs = vec![String::from("f"), String::from("f")];
    let idx = find_idx(&strs);
    let p = longest_common_prefix(strs);
    println!("idx: {:?}", idx);
    println!("p: {:?}", p);
}