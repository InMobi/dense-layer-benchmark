/*
 * Copyright (c) 2025 InMobi
 *
 * This file is part of dense-layer-benchmark.
 *
 * Licensed under the MIT License. See LICENSE file in the project root for full license information.
 */

package benchmark;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import java.util.concurrent.TimeUnit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;
import java.util.Random;
import org.ejml.data.FMatrixRMaj;
import org.ejml.dense.row.CommonOps_FDRM;
import org.openjdk.jmh.infra.Blackhole;

// Benchmark class for comparing dense layer implementations using different libraries and approaches
@BenchmarkMode({Mode.AverageTime, Mode.Throughput})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(2)
public class DenseLayerBenchmark {

    // Parameter for input/output size pairs, used to benchmark different layer sizes
    @Param({"64x32", "128x64", "256x128", "512x256", "1024x512"})
    private String sizePair = "64x32"; // Default size pair

    // Number of input and output neurons for the dense layer
    private int inSize, outSize;

    // Arrays for input data, bias, and flattened weights (row-major)
    private float[] input, bias, flatWeights;

    private FMatrixRMaj rmajWeights, rmajInput, rmajBias, rmajOutput;

    // DL4J dense layer instance
    private DenseLayer dl4jLayer;

    // Workspace manager for DL4J (no workspaces used here)
    private static final LayerWorkspaceMgr workspaceMgr = LayerWorkspaceMgr.noWorkspaces();

    // Preferred vector species for Java Vector API (auto-selects best SIMD width)
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    // Setup method to initialize all data structures before each benchmark iteration
    @Setup(Level.Iteration)
    public void setup() {
        Random rand = new Random(123);
        String[] parts = sizePair.split("x");
        inSize = Integer.parseInt(parts[0]);
        outSize = Integer.parseInt(parts[1]);

        input = new float[inSize];
        bias = new float[outSize];
        for (int i = 0; i < inSize; i++) input[i] = rand.nextFloat();
        for (int i = 0; i < outSize; i++) bias[i] = rand.nextFloat();
        float[][] weights = new float[outSize][inSize];
        for (int i = 0; i < outSize; i++)
            for (int j = 0; j < inSize; j++)
                weights[i][j] = rand.nextFloat();

        flatWeights = ArrayUtil.flatten(weights); // (outSize × inSize)

        // Initialize FMatrixRMaj versions
        // Setup for EJML dense layer computation
        rmajInput = new FMatrixRMaj(inSize, 1);
        rmajInput.setData(input);

        rmajWeights = new FMatrixRMaj(outSize, inSize);
        rmajWeights.setData(flatWeights);

        rmajBias = new FMatrixRMaj(outSize, 1);
        rmajBias.setData(bias);

        rmajOutput = new FMatrixRMaj(outSize, 1); // Reuse for output

        // Initialize DL4J dense layer
        INDArray ndInput = Nd4j.create(input, 1, inSize);
        INDArray ndWeights = Nd4j.create(weights);
        ndWeights = ndWeights.transpose(); // (inSize × outSize)
        INDArray ndBias = Nd4j.create(bias, 1, outSize);

        org.deeplearning4j.nn.conf.layers.DenseLayer denseLayerConf =
            new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                .nIn(inSize)
                .nOut(outSize)
                .activation(Activation.IDENTITY)
                .build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                                           .layer(denseLayerConf)
                                           .build();

        dl4jLayer = (DenseLayer) denseLayerConf.instantiate(conf, null, 0, Nd4j.create(DataType.FLOAT,
            (long) inSize * outSize + outSize), false, DataType.FLOAT);
        dl4jLayer.setParam("W", ndWeights);
        dl4jLayer.setParam("b", ndBias);
        dl4jLayer.setInput(ndInput, workspaceMgr);
    }

    @Benchmark
    public float[] scalarDense(Blackhole blackhole) {
        float[] output = new float[outSize];
        for (int i = 0; i < outSize; i++) {
            float sum = bias[i];
            int offset = i * inSize;
            for (int j = 0; j < inSize; j++) {
                sum += flatWeights[offset + j] * input[j];
            }
            output[i] = sum;
        }
        blackhole.consume(output);
        return output;
    }

    @Benchmark
    public float[] ejmlDense(Blackhole blackhole) {
        CommonOps_FDRM.mult(rmajWeights, rmajInput, rmajOutput);
        CommonOps_FDRM.addEquals(rmajOutput, rmajBias);
        blackhole.consume(rmajOutput.getData());
        return rmajOutput.getData();
    }

    @Benchmark
    public float[] dl4jDense(Blackhole blackhole) {
        INDArray result = dl4jLayer.activate(false, workspaceMgr);
        blackhole.consume(result);
        return result.data().asFloat();
    }

    // Benchmark method for dense layer using Java Vector API (manual reduction)
    @Benchmark
    public float[] vectorApiDense(Blackhole blackhole) {
        float[] output = new float[outSize];
        // Loop over each output neuron
        for (int i = 0; i < outSize; i++) {
            output[i] = bias[i]; // Initialize with bias
            int offset = i * inSize;
            int j = 0;
            // Vectorized computation for most of the input
            for (; j < SPECIES.loopBound(inSize); j += SPECIES.length()) {
                FloatVector vInput = FloatVector.fromArray(SPECIES, input, j); // Load input vector
                FloatVector vWeight = FloatVector.fromArray(SPECIES, flatWeights, offset + j); // Load weight vector
                output[i] += vInput.mul(vWeight).reduceLanes(VectorOperators.ADD); // Multiply and sum
            }
            // Handle remaining elements (tail case)
            for (; j < inSize; j++) {
                output[i] += flatWeights[offset + j] * input[j];
            }
        }
        blackhole.consume(output);
        return output;
    }

    // Benchmark method for dense layer using Java Vector API with FMA (Fused Multiply-Add)
    @Benchmark
    public float[] vectorApiFmaDense(Blackhole blackhole) {
        float[] output = new float[outSize];
        for (int i = 0; i < outSize; i++) {
            output[i] = bias[i];
            FloatVector acc = FloatVector.zero(SPECIES);
            int offset = i * inSize;
            int j = 0;
            for (; j < SPECIES.loopBound(inSize); j += SPECIES.length()) {
                FloatVector vInput = FloatVector.fromArray(SPECIES, input, j);
                FloatVector vWeight = FloatVector.fromArray(SPECIES, flatWeights, offset + j);
                acc = vInput.fma(vWeight, acc);
            }
            output[i] += acc.reduceLanes(VectorOperators.ADD);
            for (; j < inSize; j++) {
                output[i] += flatWeights[offset + j] * input[j];
            }
        }
        blackhole.consume(output);
        return output;
    }
}
