use std::cmp;
use std::cmp::max;

pub fn calc_memo(nums: Vec<i32>) -> Vec<i32> {
    let mut memo: Vec<i32> = vec![0; nums.len()];
    memo[nums.len()-1] = nums[nums.len()-1];
    if nums.len() <= 1 {
        return memo;
    }
    for i in (0..nums.len()-1).rev() {
        println!("i: {:?}", i);
        memo[i] = max(nums[i], nums[i]+memo[i+1]);
    }
    return memo;
}

fn main() {
    let mut memo = calc_memo(vec![-2, -1]);
    println!("memo: {:?}", &memo);
}
