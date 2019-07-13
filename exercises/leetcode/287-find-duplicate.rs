use std::iter::Sum;


fn run_find_duplicate(nums: Vec<i32>) -> i32 {
    let mut min: i32 = 1;
    let mut max: i32 = nums.len() as i32 - 1;
    while max > min {
        let mid = (min + max) / 2;
        let lcount: i32 = nums.iter().filter(|&x| x <= &mid).count() as i32;
        if lcount > mid {
            max = mid;
        } else {
            min = mid + 1;
        }
    }
   return min;
}

fn main() {
    let v = vec![1, 2, 2];
    let n = run_find_duplicate(v);
    println!("hello: {:?}", n);
}