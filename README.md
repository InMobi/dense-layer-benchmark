# Dense-layer-benchmark

This repository benchmarks the performance of various dense (fully connected) layer implementations in Java using JMH (Java Microbenchmark Harness).

### 📌 Purpose

This project aims to evaluate and compare the performance of:

✅ Scalar (naive) implementation

✅ EJML matrix multiplication

✅ DeepLearning4j (DL4J)

✅ Java Vector API

✅ Java Vector API with Fused Multiply Add (FMA)

across varying input and output sizes, and multi-threaded environments.

### 🚀 Benchmarked Implementations

| Implementation | Description                                                          |
|----------------|----------------------------------------------------------------------|
| Scalar         | Plain nested loop multiplication with bias addition                  |
| EJML           | Matrix multiplication using EJML's FMatrixRMaj and CommonOps_FDRM    |
| DL4J           | Dense layer using DL4J with identity activation                      |
| Vector         | SIMD-based computation using Java Vector API                         |
| FMA            | SIMD-based computation using Java Vector API with Fused Multiply Add |

### 📊 Benchmark Modes
Two benchmark modes are evaluated:

AverageTime – time per operation (μs)

Throughput – operations per second (ops/µs)

### ✅ Sanity Test
A sanity test is included to ensure correctness across all implementations. It verifies that outputs are numerically consistent (within float tolerance).

### 🧪 How to Run
#### Build the Project
<pre> <code> mvn clean install </code> </pre>

#### Run the benchmark
<pre> <code>
java --enable-preview --add-modules jdk.incubator.vector -jar target/benchmark.jar \
-t 8 -rf csv -rff results_threads8.csv
</code> </pre>

You can also customize threads, benchmark mode and output format.
     
