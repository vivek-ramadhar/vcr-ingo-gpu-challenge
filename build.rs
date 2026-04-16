use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
    println!("cargo:rerun-if-changed=./cuda");

    let arch_type = env::var("ARCH_TYPE").unwrap_or(String::from("native"));
    let mut arch = String::from("-arch=");
    arch.push_str(&arch_type);

    let kernel = env::var("CUDA_KERNEL").unwrap_or_else(|_| String::from("mult"));
    let cu_path = format!("./cuda/{}.cu", kernel);
    println!("cargo:rerun-if-env-changed=CUDA_KERNEL");
    println!("cargo:rerun-if-changed={}", cu_path);

    let mut nvcc = cc::Build::new();

    nvcc.cuda(true);
    nvcc.debug(false);
    nvcc.flag(&arch);
    nvcc.files([cu_path.as_str()]);
    nvcc.compile("ingo_challenge");
}
