use std::env;


mod vivien;
mod vikram;



fn main() {

    let input: Vec<String> = env::args().collect();

    match input[1].as_str() {

        "vivien" => vivien::unbound_vivien(),
        "vikram" => vikram::unbound_vikram(),
        _ => println!("Unrecognized argument: {}. Use 'vivien' or 'vikram'", input[1]),
    }

}
