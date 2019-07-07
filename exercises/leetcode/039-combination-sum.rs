fn combination_sum(candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
}

fn recursive_combination_sum(candidates: &[i32], target: i32) -> Vec<i32> {
    for i in 0..candidates.len() {
        n = target - candidates[i];
        if n == 0 {
            candidates
        } else if n > 0 {
            let vs = recursive_combination_sum(candidates, n);
            v.add(candidates[i]);
            return v
        }
    }
}

fn main() {
}
