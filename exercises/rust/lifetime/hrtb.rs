use std::io::Read;

trait Checksum<R:Read> {
    fn calc(&mut self, r:R) -> Vec<u8>;
}

struct Xor;

impl <R:Read> Checksum<R> for Xor {
    fn calc(&mut self, mut r:R) -> Vec<u8> {
        let mut res: u8 = 0;
        let mut buf = [0u8;8];
        loop {
            let read = r.read(&mut buf).unwrap();
            if read == 0 { break }
            for b in &buf[..read] {
                res ^= b;
            }
        }
        
        vec![res]
    }
}

fn calc_checksum<'a>(buf: &'a [u8], mut c: impl Checksum<&'a [u8]>) -> Vec<u8> {
    c.calc(buf)
}

fn calc_file_with_checksum(_path: String, mut checksumer: impl for<'a> Checksum<&'a [u8]>) -> Vec<u8> {
    let buf = "blah blah blah".to_string().into_bytes();
    checksumer.calc(&buf)
}

fn main(){
    let c = calc_file_with_checksum("/tmp/foo".to_string(), Xor);
    println!("val: {:?}", c)
}
