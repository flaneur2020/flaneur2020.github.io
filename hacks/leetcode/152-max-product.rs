use std::cmp::max;

fn find_max_product(nums: Vec<i32>) -> i32 {
    let mut max_product = std::i32::MIN;
    for i in 0..nums.len() {
        let mut j = i;
        let mut p = 1;
        while j < nums.len() {
            p *= nums[j];
            max_product = max(p, max_product);
            j += 1;
        }
    }
    return max_product;
}

fn main() {
    let v = vec![-2, 0, -1];
    let m = find_max_product(v);
    println!("max product: {:?}", m);
}