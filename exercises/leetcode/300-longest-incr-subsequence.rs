use std::cmp::max;

fn longest_subsequence(nums: Vec<i32>, memo: &mut Vec<i32>) {
    for i in (0..nums.len()-1).rev() {
        let mut longest = 0;
        for j in i+1..nums.len() {
            if nums[i] < nums[j] {
                longest = max(longest, memo[j]);
            }
        }
        memo[i] = longest + 1;
    }
}


fn main() {
    // let nums = vec![10, 9, 2, 5, 3, 7, 101, 18];
    let nums = vec![2, 2];
    // let memo = vec![2, 2, 4, 3, 3, 2, 1, 1];
    let mut memo = vec![1; nums.len()];
    longest_subsequence(nums, &mut memo);
    let max = memo.iter().fold(0, |s, v| max(s, *v));
    println!("memo: {:?}", max);
}