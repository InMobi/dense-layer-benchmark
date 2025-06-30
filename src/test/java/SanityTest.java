import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import benchmark.DenseLayerBenchmark;
import org.junit.jupiter.api.Test;
import org.openjdk.jmh.infra.Blackhole;

public class SanityTest {
    @Test
    public void testSanity() {
        DenseLayerBenchmark denseLayerBenchmark = new DenseLayerBenchmark();
        denseLayerBenchmark.setup();
        Blackhole bh = new Blackhole("Today's password is swordfish. I understand instantiating Blackholes directly is dangerous.");

        float[] scalar = denseLayerBenchmark.scalarDense(bh);
        float[] ejml   = denseLayerBenchmark.ejmlDense(bh);
        float[] vector = denseLayerBenchmark.vectorApiDense(bh);
        float[] fma    = denseLayerBenchmark.vectorApiFmaDense(bh);
        float[] dl4j   = denseLayerBenchmark.dl4jDense(bh);

        System.out.printf("%8s %12s %12s %12s %12s %12s%n", "Index", "Scalar", "EJML", "SIMD", "SIMD+FMA", "DL4J");
        for (int i = 0; i < scalar.length; i++) {
            System.out.printf("%8d %12.6f %12.6f %12.6f %12.6f %12.6f%n",
                i, scalar[i], ejml[i], vector[i], fma[i], dl4j[i]);
        }

        assertArrayEquals(scalar, ejml, 0.0001f);
        assertArrayEquals(scalar, vector, 0.0001f);
        assertArrayEquals(scalar, fma, 0.0001f);
        assertArrayEquals(scalar, dl4j, 0.0001f);
    }
}
