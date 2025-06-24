# dense-layer-benchmark

This repository benchmarks the performance of various dense (fully connected) layer implementations in Java using JMH (Java Microbenchmark Harness).

📌 Purpose
This project aims to evaluate and compare the performance of:

✅ Scalar (naive) implementation

✅ EJML matrix multiplication

✅ DeepLearning4j (DL4J)

✅ Java Vector API

✅ Java Vector API with Fused Multiply Add (FMA)

across varying input and output sizes, and multi-threaded environments.

🚀 Benchmarked Implementations
Implementation	Description
Scalar	Plain nested loop multiplication with bias addition
EJML	Matrix multiplication using EJML's FMatrixRMaj and CommonOps_FDRM
DL4J	Dense layer using DL4J with identity activation
Vector	SIMD-based computation using Java Vector API
FMA	SIMD-based computation using Java Vector API with Fused Multiply Add

⚙️ Benchmark Setup
Benchmarks are written using JMH

Runs are configured with multiple threads: 1, 8, 16, 32

Benchmarked on:

Machine type: n2d-standard-32 (32 vCPUs, 128 GB RAM)

CPU: AMD Milan (x86_64)

OS: Ubuntu

Warmup & Measurement Configuration
java
Copy
Edit
@Warmup(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Measurement(iterations = 5, time = 500, timeUnit = TimeUnit.MILLISECONDS)
@Fork(1)
📊 Benchmark Modes
Two benchmark modes are evaluated:

AverageTime – time per operation (μs)

Throughput – operations per second (ops/s)

✅ Sanity Test
A sanity test is included to ensure correctness across all implementations. It verifies that outputs are numerically consistent (within float tolerance).

🧪 How to Run
Build the Project
bash
Copy
Edit
mvn clean install
Run Benchmarks
bash
Copy
Edit
java --add-modules jdk.incubator.vector -jar target/benchmark.jar
You can also customize threads and output format:

bash
Copy
Edit
java --add-modules jdk.incubator.vector -jar target/benchmark.jar \
     -t 8 -bm avgt -foe true -rf csv -rff results_threads8.csv
     
📚 Blog
📝 Read the full blog:
"Leveraging Java Vector APIs to Optimize Neural Network Inferencing"
[Link to blog goes here]
