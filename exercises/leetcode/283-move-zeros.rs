
fn move_zeroes(nums: &mut Vec<i32>) {
    let mut i = 0;
    while i < nums.len() - 1 {
        if nums[i] == 0 {
            let mut j = i + 1;
            while nums[j] == 0 && j < nums.len() - 1 {
                j += 1;
            }
            let tmp = nums[j];
            nums[j] = nums[i];
            nums[i] = tmp;
        }
        i += 1;
    }
}

fn main() {
    let mut v = vec![0, 1, 0, 3, 12];
    move_zeroes(&mut v);
    println!("{:?}", v);
}