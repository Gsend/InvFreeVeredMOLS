//! Superseded by three per-algorithm demos.  Run those instead:
//!
//!     cargo run --example demo_alg1_modified_cholesky --release
//!     cargo run --example demo_alg2_sgso              --release
//!     cargo run --example demo_alg3_weighted_gi       --release
//!
//! Each demo simulates a 300×100 linear system with noise, runs the
//! corresponding olssm algorithm, and compares it against a textbook
//! Householder-QR reference solution.

fn main() {
    println!();
    println!("This combined demo has been superseded by three per-algorithm demos.");
    println!("Run instead:");
    println!("    cargo run --example demo_alg1_modified_cholesky --release");
    println!("    cargo run --example demo_alg2_sgso              --release");
    println!("    cargo run --example demo_alg3_weighted_gi       --release");
    println!();
}
