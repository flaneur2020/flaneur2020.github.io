
fn combine_impl(nums: &[i32], k: i32) -> Vec<Vec<i32>> {
    if nums.len() == 0 {
        return vec![];
    }

    if nums.len() == 1 {
        return vec![nums.to_vec(), vec![]];
    }

    if k == 0 {
        return vec![];
    }

    let mut results: Vec<Vec<i32>> = vec![];
    for sub_slice in combine_impl(&nums[1..nums.len()], k-1) {
        let mut r: Vec<i32> = vec![nums[0]];
        r.extend(sub_slice);
        results.push(r);
    }

    for sub_slice in combine_impl(&nums[1..nums.len()], k) {
        results.push(sub_slice);
    }
    return results;
}

fn main() {
    let v = vec![1, 2, 3, 4];
    let r = combine_impl(&v, 2);
    println!("hello: {:?}", r);
}
