fn run_increasing_triplet(nums: Vec<i32>) -> bool {  
    let mut min: i32 = std::i32::MAX;
    let mut mid: i32 = std::i32::MAX;
    for num in nums {
		if num < min {
			min = num;
		}
		if num > min {
			mid = std::cmp::min(mid, num);
		}
		if num > mid {
			return true;
		}
    }
	return false
}

fn main() {
    // let nums = vec![7, 3, 8, 4, 5, 2, 1];
	let nums = vec![0,6,3,-1,2,4,1];
    let r = run_increasing_triplet(nums);
	println!("r: {:?}\n", r);
}