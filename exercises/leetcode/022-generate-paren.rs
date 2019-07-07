fn recursive_gen(result: &mut Vec<String>, s: String, l: usize, r: usize, n: usize) {
    if l > n || r > n || r > l {
        return;
    }

    if l == n && r == n {
        result.push(s);
        return;
    }

    recursive_gen(result, s.clone() + "(", l+1, r, n);
    recursive_gen(result, s.clone() + ")", l, r+1, n);
}


fn generate_parens(n: usize) -> Vec<String> {
    let mut rs: Vec<String> = vec![];
    let s: String = "".to_string();
    recursive_gen(&mut rs, s, 0, 0, n);
    return rs
}

fn main() {
    let rs: Vec<String> = generate_parens(3);
    println!("parens: {:?}", rs)
}
