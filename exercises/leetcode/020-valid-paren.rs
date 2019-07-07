fn valid_paren(s: String) -> bool {
    let buf: &[u8] = s.as_bytes();
    let mut stack: Vec<u8> = vec![];
    for i in 0..buf.len() {
        if buf[i] == '{' as u8 || buf[i] == '[' as u8 || buf[i] == '(' as u8 {
            stack.push(buf[i]);
        } else {
            if stack.len() == 0 {
                return false;
            }
            let st = stack.pop().unwrap();
            let mismatch = buf[i] != match st as char {
                '{' => '}' as u8,
                '[' => ']' as u8,
                '(' => ')' as u8,
                _ => 'x' as u8,
            };
            if mismatch {
                return false
            }
        }
    }
    return stack.len() == 0;
}

fn main() {
    println!("{:?}", valid_paren("[()()]".to_string()));
    println!("{:?}", valid_paren("[()]()]".to_string()));
}