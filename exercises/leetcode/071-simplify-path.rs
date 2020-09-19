fn simplify_path(path: String) -> String {
    let chunks = path.split('/');
    let mut stack: Vec<String> = vec![];
    for chunk in chunks {
        if chunk == "." || chunk == "" {
            continue;
        } else if chunk == ".." {
            stack.pop();
        } else {
            stack.push(chunk.to_string());
        }
    }
    return ["/".to_string(), stack.join("/")].concat();
}

fn main() {
    println!("{:?}", simplify_path("/home/".to_string()));
    println!("{:?}", simplify_path("/../".to_string()));
    println!("{:?}", simplify_path("/home//foo/".to_string()));
    println!("{:?}", simplify_path("/a/./b/../../c/".to_string()));
    println!("{:?}", simplify_path("/a/../../b/../c//.//".to_string()));
}
