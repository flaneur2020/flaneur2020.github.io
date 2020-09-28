fn transform(s: &[u8], num_rows: usize) -> Vec<Vec<u8>> {
    let chunk_size = num_rows + num_rows - 2;
    let chunk_cols = num_rows - 1;
    let num_cols: usize = ((s.len() as f64) / chunk_size as f64).ceil() as usize * chunk_cols;
    let mut matrix = vec![vec![0 as u8; num_cols]; num_rows];
    for i in 0..s.len() {
        let is_vertical: bool = i % chunk_size < num_rows;
        if is_vertical {
            let col = (i as f64 / chunk_size as f64).floor() as usize * (num_rows - 1);
            let row = i % chunk_size;
            println!("row: {}, col: {}, s[i]: {}, i: {}", row, col, s[i] as char, i);
            matrix[row][col] = s[i];
        } else {
            let row = (num_rows - 1) - (i % chunk_size - num_rows + 1);
            let col = (i as f64 / chunk_size as f64).floor() as usize * chunk_cols + i % chunk_size - num_rows + 1;
            matrix[row][col] = s[i];
        }
    }
    return matrix;
}

fn main() {
    let m = transform("LEETCODEISHIRING".as_bytes(), 3);
    let sv: Vec<String> = m.into_iter().map(|v| String::from_utf8_lossy(&v).to_string()).collect();
    println!("{:?}", sv);
}