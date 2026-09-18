fn main() {
    let command = std::env::args().nth(1).unwrap_or_else(|| "doctor".into());
    match command.as_str() {
        "doctor" => println!("PTR control plane scaffold: OK"),
        "layout" => println!("See docs/architecture and README.md"),
        _ => eprintln!("usage: ptrctl [doctor|layout]"),
    }
}
