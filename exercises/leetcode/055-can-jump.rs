// dp: 递归回溯
fn recursive_can_jump(nums: &[i32], idx: usize, memo: &mut [i32]) -> bool {
    if nums.len() <= idx + 1 {
        return true;
    }
    let n = nums[idx];
    if (n as usize) >= nums.len() - 1 {
        return true;
    }
    if memo[idx] == 1 {
        return false;
    }
    for i in (1..n+1).rev() {
        let nidx = idx + (i as usize);
        let ok = recursive_can_jump(&nums, nidx, memo);
        if ok {
            return true;
        }
    }
    memo[idx] = 1;
    return false;
}

// dp: 从底向上
fn dp(nums: &[i32]) -> bool {
    let mut memo = vec![0; nums.len()];
    memo[(nums.len()-1) as usize] = 1;
    for i in (0..nums.len()-1).rev() {
        for j in (1..((nums[i] + 1) as usize)).rev() {
            if (i+j as usize) >= nums.len() {
                memo[i] = 1;
            } else if memo[i+j] == 1 {
                memo[i] = 1;
            }
        }
    }
    println!("memo: {:?}", memo);
    return memo[0] == 1;
}

fn main() {
    let v = vec![2, 0];
    let mut memo = vec![0; v.len()];
    // let r = recursive_can_jump(&v, 0, &mut memo);
    let r = dp(&v);
    println!("{:?}", r);
}
