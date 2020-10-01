fn transform(s: Vec<u8>, num_rows: usize) -> Vec<Vec<u8>> {
    let chunk_size = num_rows + num_rows - 2;
    let chunk_cols = num_rows - 1;
    let num_cols: usize = ((s.len() as f64) / chunk_size as f64).ceil() as usize * chunk_cols;
    let mut matrix = vec![vec![' ' as u8; num_cols]; num_rows];
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

fn transform_to_string(s: String, num_rows: usize) -> String {
    if num_rows == 1 {
        return s;
    }
    let m: Vec<Vec<u8>> = transform(s.into_bytes(), num_rows);
    let buf: Vec<u8> = m.into_iter().flatten().collect();
    let filtered_buf: Vec<u8> = buf.into_iter().filter(|c| *c != ' ' as u8).collect();
    return String::from_utf8_lossy(&filtered_buf).to_string();
}

fn main() {
    let mut r: String;
    
    r = transform_to_string(String::from("LEETCODEISHIRING"), 4);
    println!("{}", r);

    r = transform_to_string(String::from("A"), 1);
    println!("{}", r);
}