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

fn main() {
    let v = vec![1,1,1,0];
    let mut memo = vec![0; v.len()];
    let r = recursive_can_jump(&v, 0, &mut memo);
    println!("{:?}", r);
    println!("{:?}", memo);
}
