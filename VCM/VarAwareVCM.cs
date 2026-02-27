using Microsoft.VisualBasic;

namespace VarAwareVCM;

public abstract class CorrectionFactorsBase
{
    public abstract void StartIteration();
    public abstract void EndIteration(uint iteration, Scene scene);
    public virtual float Get(int cameraPathEdges, int lightEdges, int totalEdges, Pixel pixel) { return 1.0f; }
    public virtual float Get(int cameraPathEdges, int totalEdges, Pixel pixel) { return 1.0f; }
    public virtual void WriteToFiles(string basename) { }
    public virtual void WriteFactors(string basename, int mindepth, int maxdepth) { }
    public virtual void WriteMoments(string basename) { }
    public virtual void Add(int cameraPathEdges, int lightPathEdges, int totalEdges,
                    Pixel pixel, RgbColor value, float kernelWeight, float actualMisweight, float mergeMisSum, float lightTracerMis, float nextEvtMis)
    { }

    public virtual void Add(int cameraPathEdges, int lightPathEdges, int totalEdges,
                Pixel pixel, RgbColor value, float kernelWeight, float actualMisweight, float mergeMisSum, Span<float> mergeMis)
    { }

    public virtual void Add(int cameraPathEdges, int lightPathEdges, int totalEdges,
                Pixel pixel, RgbColor value, float kernelWeight, float actualMisweight, float mergeMisSum)
    { }

    public virtual bool IsReady()
    {
        return false;
    }
}

public class VarAwareVCM : VertexConnectionAndMerging {

    protected VarAwareMergingFactors mergeFactors;
    protected VarAwareLightTracerFactors lightTracerFactors;
    protected VarAwareNextEvtFactors nextevtFactors;
    protected VarAwareHitFactors hitFactors;


    public int NumTrainSamples = 0;
    public bool ReadFromPrecompute = false;
    public bool UseFilteredFactors = false;
    public bool EnableLightTracerFactor = false;
    public bool EnableNextEvtFactor = false;
    public bool EnableHitFactor = false;

    public class VarAwareMergingFactors {
        public VarAwareMergingFactors(int maxDepth, int width, int height, bool useFilteredFactors, int numTrainSamples) {
            moments = new(maxDepth - 1);
            pixelValues = new(maxDepth - 1);
            iterationEstimates = new(maxDepth - 1);
            iterationMoments = new(maxDepth - 1);
            varianceFactors = new(maxDepth - 1);
            variances = new(maxDepth - 1);
            for (int len = 2; len <= maxDepth; ++len) { // all depths with correlated merges (i.e., no DI)
                moments.Add(new(len - 1));
                variances.Add(new(len - 1));
                pixelValues.Add(new(len - 1));
                iterationEstimates.Add(new(len - 1));
                iterationMoments.Add(new(len - 1));
                varianceFactors.Add(new(len - 1));
                for (int i = 1; i < len; ++i) { // all merges with correlation for paths of length "len"
                    moments[^1].Add(new(width, height));
                    pixelValues[^1].Add(new(width, height));
                    iterationEstimates[^1].Add(new(width, height));
                    iterationMoments[^1].Add(new(width, height));
                    varianceFactors[^1].Add(new(width, height));
                    variances[^1].Add(new(width, height));
                    variances[^1][^1].Fill(float.MaxValue);
                }
            }

            // switch between accurate and filtered factors. The later only need 1 iter to estimate.
            this.UseFilteredFactors = useFilteredFactors;
            this.numTrainSamples = numTrainSamples;
        }

        /// <summary>
        /// Read precomputed variance factors
        /// </summary>
        /// <param name="maxDepth"></param>
        /// <param name="basename"></param>
        public VarAwareMergingFactors(int maxDepth, string basename, bool useFilteredFactors)
        {
            varianceFactors = new(maxDepth - 1);
            for (int len = 2; len <= maxDepth; ++len)
            { // all depths with correlated merges (i.e., no DI)
                varianceFactors.Add(new(len - 1));
                for (int i = 1; i < len; ++i)
                { // all merges with correlation for paths of length "len"
                    var filename = $"{basename}-depth-{len}-merge-{i}.exr";
                    if (useFilteredFactors)
                        filename = $"{basename}-depth-{len}-merge-{i}-filtered.exr";
                    varianceFactors[^1].Add(new MonochromeImage(filename));
                }
            }
            this.UseFilteredFactors = useFilteredFactors;
            isReady = true;
        }

        public void StartIteration() {
            curIteration++;

            if (isReady) return;

            // Scale values of the previous iteration to account for having more samples
            if (curIteration > 1) {
                Parallel.For(0, moments.Count, i => {
                    for (int k = 0; k < moments[i].Count; ++k) {
                        moments[i][k].Scale((curIteration - 1.0f) / curIteration);
                        pixelValues[i][k].Scale((curIteration - 1.0f) / curIteration);
                        iterationMoments[i][k].Scale((curIteration - 1.0f) / curIteration);

                        iterationEstimates[i][k].Scale(0);
                    }
                });
            }
        }

        public void EndIteration(uint iteration, Scene scene)
        {
            if (isReady) return;

            if (UseFilteredFactors)
                ComputeFilteredFactors();
            else
                ComputeAccurateFactors();

            if (iteration == numTrainSamples - 1)
                isReady = true;
        }

        public void Add(int cameraPathEdges, int lightPathEdges, int totalEdges,
                        Pixel pixel, RgbColor value) {
            if (isReady) return;
            bool isMerge = lightPathEdges > 0 && lightPathEdges + cameraPathEdges == totalEdges;
                if (isMerge && cameraPathEdges > 0) { // Primary merges have zero covariance
                float v = value.Average;
                moments[totalEdges - 2][cameraPathEdges - 1].
                    AtomicAdd(pixel.Col, pixel.Row, v * v / curIteration);
                pixelValues[totalEdges - 2][cameraPathEdges - 1].
                    AtomicAdd(pixel.Col, pixel.Row, v / curIteration);
                iterationEstimates[totalEdges - 2][cameraPathEdges - 1].
                    AtomicAdd(pixel.Col, pixel.Row, v);
            }
        }

        public float Get(int cameraPathEdges, int totalEdges, Pixel pixel) {
            if (!isReady) return 1.0f;
            if (cameraPathEdges < 1) return 1.0f;
            if (totalEdges < 2) return 1.0f;

            return varianceFactors[totalEdges - 2][cameraPathEdges - 1]
                .GetPixel(pixel.Col, pixel.Row);
        }

        public void WriteToFiles(string basename) {
            if (!isReady) return;

            for (int i = 0; i < varianceFactors.Count; ++i) {
                for (int k = 0; k < varianceFactors[i].Count; ++k) {
                    var filename = $"{basename}-depth-{i+2}-merge-{k+1}.exr";
                    if(UseFilteredFactors)
                        filename = $"{basename}-depth-{i+2}-merge-{k+1}-filtered.exr";
                    varianceFactors[i][k].WriteToFile(filename, 1);
                }
            }
        }

        void ComputeFilteredFactors()
        {
            int width = moments[0][0].Width;
            int height = moments[0][0].Height;
            MonochromeImage momentBuffer = new(width, height);
            MonochromeImage varianceBuffer = new(width, height);
            int blurRadius = 4;

            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < moments.Count; ++i)
            {
                for (int k = 0; k < moments[i].Count; ++k)
                {
                    // Estimate the pixel variances:
                    // First, we blur the image. Then, we subtract the blurred version from the original.
                    // Finally, we compute and square the difference, multiplying by the number of iterations
                    // to obtain a coarse estimate of the variance in a single iteration.
                    Filter.RepeatedBox(pixelValues[i][k], varianceBuffer, blurRadius);
                    Parallel.For(0, height, row => {
                        for (int col = 0; col < width; ++col)
                        {
                            var value = pixelValues[i][k].GetPixel(col, row);
                            var delta = value - varianceBuffer.GetPixel(col, row);
                            var variance = delta * delta * curIteration;
                            varianceBuffer.SetPixel(col, row, variance);
                        }
                    });
                    Filter.RepeatedBox(varianceBuffer, varianceFactors[i][k], blurRadius);
                    
                    // Also filter the second moment estimates
                    Filter.RepeatedBox(moments[i][k], momentBuffer, blurRadius);

                    // Compute the ratio for all non-zero pixels
                    Parallel.For(0, height, row => {
                        for (int col = 0; col < width; ++col)
                        {
                            var variance = varianceFactors[i][k].GetPixel(col, row);
                            var moment = momentBuffer.GetPixel(col, row);
                            if (variance > 0 && moment > 0)
                            {
                                varianceBuffer.SetPixel(col, row, moment / variance);
                                variances[i][k][col, row] = variance;
                            }
                            else
                            {
                                varianceBuffer.SetPixel(col, row, 1);
                            }
                        }
                    });

                    // Apply a wide filter to the ratio image, too
                    Filter.RepeatedBox(varianceBuffer, varianceFactors[i][k], blurRadius);
                }
            }
        }

        void ComputeAccurateFactors()
        {
            int width = moments[0][0].Width;
            int height = moments[0][0].Height;

            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < moments.Count; ++i)
            {
                for (int k = 0; k < moments[i].Count; ++k)
                {
                    Parallel.For(0, height, row => {
                        for (int col = 0; col < width; ++col)
                        {
                            var iterVal = iterationEstimates[i][k][col, row];
                            iterationMoments[i][k][col, row] += iterVal * iterVal / curIteration;
                            var value = pixelValues[i][k][col, row];
                            var variance = iterationMoments[i][k][col, row] - value * value;

                            var moment = moments[i][k][col, row];
                            if (variance > 0 && moment > 0)
                                varianceFactors[i][k][col, row] = moment / variance;
                            else
                                varianceFactors[i][k][col, row] = 1;
                        }
                    });
                }
            }
        }


        public bool isReady = false;
        bool UseFilteredFactors = false;
        public int numTrainSamples = 0;
        int curIteration = 0;
        List<List<MonochromeImage>> moments;
        List<List<MonochromeImage>> pixelValues;
        List<List<MonochromeImage>> iterationEstimates;
        List<List<MonochromeImage>> iterationMoments;
        List<List<MonochromeImage>> varianceFactors;
        List<List<MonochromeImage>> variances;
    }

    public class VarAwareHitFactors
    {
        public VarAwareHitFactors(int maxDepth, int width, int height, bool UseFiltered, int numTrainSamples, bool enablefactor)
        {
            this.Enablefactor = enablefactor;
            if (!enablefactor) return;
            moments = new(maxDepth - 1);
            pixelValues = new(maxDepth - 1);
            varianceFactors = new(maxDepth - 1);
            iterationEstimates = new(maxDepth - 1);
            iterationMoments = new(maxDepth - 1);
            for (int i = 1; i < maxDepth; ++i)
            { // all depths with correlated next event (i.e., no DI)
                moments.Add(new(width, height));
                pixelValues.Add(new(width, height));
                varianceFactors.Add(new(width, height));
                iterationEstimates.Add(new(width, height));
                iterationMoments.Add(new(width, height));


            }

            this.numTrainSamples = numTrainSamples;
            this.UseFilteredFactors = UseFiltered;
        }

        public VarAwareHitFactors(int maxDepth, string basename, bool UseFiltered, bool enablefactor)
        {
            this.Enablefactor = enablefactor;
            if (!enablefactor) return;
            varianceFactors = new(maxDepth - 1);
            for (int i = 2; i <= maxDepth; ++i)
            { // all depths with correlated next event (i.e., no DI)

                var filename = Path.Join(basename, $"variance-factors-nextevt-{i}.exr");
                if (UseFiltered)
                    filename = Path.Join(basename, $"variance-factors-nextevt-{i}-filtered.exr");
                varianceFactors.Add(new MonochromeImage(filename));
                isReady = true;
            }
        }

        public void StartIteration()
        {
            if (!Enablefactor) return;

            curIteration++;
            if (isReady) return;
            // Scale values of the previous iteration to account for having more samples
            if (curIteration > 1)
            {
                for (int i = 0; i < moments.Count; ++i)
                {
                    pixelValues[i].Scale((curIteration - 1.0f) / curIteration);

                    moments[i].Scale((curIteration - 1.0f) / curIteration);
                    iterationMoments[i].Scale((curIteration - 1.0f) / curIteration);

                    iterationEstimates[i].Scale(0);
                }
            }
        }

        public void EndIteration(uint iteration, Scene scene)
        {
            if (!Enablefactor) return;
            if (isReady) return;

            if (UseFilteredFactors)
                ComputeFilteredFactors();
            else
                ComputeAccurateFactors();

            if (iteration == numTrainSamples - 1)
            {
                isReady = true;
            }
        }

        void ComputeFilteredFactors()
        {
            int width = moments[0].Width;
            int height = moments[0].Height;
            MonochromeImage momentBuffer = new(width, height);
            MonochromeImage varianceBuffer = new(width, height);
            int blurRadius = 4;

            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < moments.Count; ++i)
            {
                {
                    // Estimate the pixel variances:
                    // First, we blur the image. Then, we subtract the blurred version from the original.
                    // Finally, we compute and square the difference, multiplying by the number of iterations
                    // to obtain a coarse estimate of the variance in a single iteration.
                    SimpleImageIO.Filter.RepeatedBox(pixelValues[i], varianceBuffer, blurRadius);
                    Parallel.For(0, height, row => {
                        for (int col = 0; col < width; ++col)
                        {
                            var value = pixelValues[i].GetPixel(col, row);
                            var delta = value - varianceBuffer.GetPixel(col, row);
                            var variance = delta * delta * curIteration;
                            varianceBuffer.SetPixel(col, row, variance);
                        }
                    });
                    SimpleImageIO.Filter.RepeatedBox(varianceBuffer, varianceFactors[i], blurRadius);

                    // Also filter the second moment estimates
                    SimpleImageIO.Filter.RepeatedBox(moments[i], momentBuffer, blurRadius);

                    // Compute the ratio for all non-zero pixels
                    Parallel.For(0, height, row => {
                        for (int col = 0; col < width; ++col)
                        {
                            var variance = varianceFactors[i].GetPixel(col, row);
                            var moment = momentBuffer.GetPixel(col, row);
                            if (variance > 0 && moment > 0)
                            {
                                varianceBuffer.SetPixel(col, row, moment / variance);
                            }
                            else
                            {
                                varianceBuffer.SetPixel(col, row, 1);
                            }
                        }
                    });

                    // Apply a wide filter to the ratio image, too
                    SimpleImageIO.Filter.RepeatedBox(varianceBuffer, varianceFactors[i], blurRadius);
                }
            }
        }

        void ComputeAccurateFactors()
        {
            int width = moments[0].Width;
            int height = moments[0].Height;

            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < moments.Count; ++i)
            {
                Parallel.For(0, height, row =>
                {
                    for (int col = 0; col < width; ++col)
                    {
                        var iterVal = iterationEstimates[i][col, row];
                        iterationMoments[i][col, row] += iterVal * iterVal / curIteration;

                        var value = pixelValues[i][col, row];
                        var variance = iterationMoments[i][col, row] - value * value;
                        var moment = moments[i][col, row];


                        if (variance > 0 && moment > 0)
                            varianceFactors[i][col, row] = moment / variance;
                        else
                            varianceFactors[i][col, row] = 1;
                    }
                });
            }
        }

        public void Add(int cameraPathEdges, int lightPathEdges, int totalEdges,
                        Pixel filmPoint, RgbColor value)
        {
            if (!Enablefactor) return;
            if (isReady) return;
            bool isHit = cameraPathEdges == totalEdges;
            //if (isNextEvent && totalEdges >=2)
            if (isHit && totalEdges == 2)
            {
                moments[totalEdges - 2].AtomicAdd(filmPoint.Col, filmPoint.Row,
                    (value * value / curIteration).Average);
                pixelValues[totalEdges - 2].AtomicAdd(filmPoint.Col, filmPoint.Row,
                    value.Average / curIteration);
                iterationEstimates[totalEdges - 2].AtomicAdd(filmPoint.Col, filmPoint.Row, value.Average);
            }
        }

        public float Get(int cameraPathEdges, int totalEdges, Pixel filmPoint)
        {
            if (!Enablefactor) return 1.0f;
            if (!isReady) return 1.0f;
            bool isHit = cameraPathEdges == totalEdges;
            if (totalEdges != 2) return 1.0f;
            return varianceFactors[totalEdges - 2].GetPixel(filmPoint.Col, filmPoint.Row);
        }

        public void WriteToFiles(string basename)
        {
            if (!Enablefactor) return;
            if (!isReady) return;
            for (int i = 0; i < varianceFactors.Count; ++i)
            {
                var filename = Path.Join(basename, $"variance-factors-hit-{i + 2}.exr");
                if (UseFilteredFactors) // write filted and accurate factors separately so that we dont override by accident
                    filename = Path.Join(basename, $"variance-factors-hit-{i + 2}-filtered.exr");
                varianceFactors[i].WriteToFile(filename);

            }
        }

        bool isReady = false;
        bool UseFilteredFactors = false;
        int numTrainSamples = 0;
        int curIteration = 0;
        bool Enablefactor = false;
        protected List<MonochromeImage> iterationEstimates;
        protected List<MonochromeImage> iterationMoments;
        protected List<MonochromeImage> moments;
        protected List<MonochromeImage> pixelValues;
        protected List<MonochromeImage> varianceFactors;
    }

    public class VarAwareLightTracerFactors
    {
        public VarAwareLightTracerFactors(int maxDepth, int width, int height, bool UseFiltered, int numTrainSamples, bool enablefactor)
        {
            this.Enablefactor = enablefactor;
            if (!enablefactor) return;
            moments = new(maxDepth - 1);
            pixelValues = new(maxDepth - 1);
            varianceFactors = new(maxDepth - 1);
            iterationEstimates = new(maxDepth - 1);
            iterationMoments = new(maxDepth - 1);
            for (int i = 1; i < maxDepth; ++i)
            { // all depths with correlated next event (i.e., no DI)
                moments.Add(new(width, height));
                pixelValues.Add(new(width, height));
                varianceFactors.Add(new(width, height));
                iterationEstimates.Add(new(width, height));
                iterationMoments.Add(new(width, height));


            }

            this.numTrainSamples = numTrainSamples;
            this.UseFilteredFactors = UseFiltered;
        }

        public VarAwareLightTracerFactors(int maxDepth, string basename, bool UseFiltered, bool enablefactor)
        {
            this.Enablefactor = enablefactor;
            if (!enablefactor) return;
            varianceFactors = new(maxDepth - 1);
            for (int i = 2; i <= maxDepth; ++i)
            { // all depths with correlated next event (i.e., no DI)

                var filename = Path.Join(basename, $"variance-factors-light-tracer-{i}.exr");
                if (UseFiltered)
                    filename = Path.Join(basename, $"variance-factors-light-tracer-{i}-filtered.exr");
                varianceFactors.Add(new MonochromeImage(filename));
                isReady = true;
            }
        }

        public void StartIteration()
        {
            if (!Enablefactor) return;

            curIteration++;
            if (isReady) return;
            // Scale values of the previous iteration to account for having more samples
            if (curIteration > 1)
            {
                for (int i = 0; i < moments.Count; ++i)
                {
                    pixelValues[i].Scale((curIteration - 1.0f) / curIteration);

                    moments[i].Scale((curIteration - 1.0f) / curIteration);
                    iterationMoments[i].Scale((curIteration - 1.0f) / curIteration);

                    iterationEstimates[i].Scale(0);
                }
            }
        }

        public void EndIteration(uint iteration, Scene scene)
        {
            if (!Enablefactor) return;
            if (isReady) return;

            if (UseFilteredFactors)
                ComputeFilteredFactors();
            else
                ComputeAccurateFactors();

            if (iteration == numTrainSamples - 1)
            {
                isReady = true;
            }
        }

        void ComputeFilteredFactors()
        {
            int width = moments[0].Width;
            int height = moments[0].Height;
            MonochromeImage momentBuffer = new(width, height);
            MonochromeImage varianceBuffer = new(width, height);
            int blurRadius = 4;

            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < moments.Count; ++i)
            {
                {
                    // Estimate the pixel variances:
                    // First, we blur the image. Then, we subtract the blurred version from the original.
                    // Finally, we compute and square the difference, multiplying by the number of iterations
                    // to obtain a coarse estimate of the variance in a single iteration.
                    SimpleImageIO.Filter.RepeatedBox(pixelValues[i], varianceBuffer, blurRadius);
                    Parallel.For(0, height, row => {
                        for (int col = 0; col < width; ++col)
                        {
                            var value = pixelValues[i].GetPixel(col, row);
                            var delta = value - varianceBuffer.GetPixel(col, row);
                            var variance = delta * delta * curIteration;
                            varianceBuffer.SetPixel(col, row, variance);
                        }
                    });
                    SimpleImageIO.Filter.RepeatedBox(varianceBuffer, varianceFactors[i], blurRadius);

                    // Also filter the second moment estimates
                    SimpleImageIO.Filter.RepeatedBox(moments[i], momentBuffer, blurRadius);

                    // Compute the ratio for all non-zero pixels
                    Parallel.For(0, height, row => {
                        for (int col = 0; col < width; ++col)
                        {
                            var variance = varianceFactors[i].GetPixel(col, row);
                            var moment = momentBuffer.GetPixel(col, row);
                            if (variance > 0 && moment > 0)
                            {
                                varianceBuffer.SetPixel(col, row, moment / variance);
                            }
                            else
                            {
                                varianceBuffer.SetPixel(col, row, 1);
                            }
                        }
                    });

                    // Apply a wide filter to the ratio image, too
                    SimpleImageIO.Filter.RepeatedBox(varianceBuffer, varianceFactors[i], blurRadius);
                }
            }
        }

        void ComputeAccurateFactors()
        {
            int width = moments[0].Width;
            int height = moments[0].Height;

            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < moments.Count; ++i)
            {
                Parallel.For(0, height, row =>
                {
                    for (int col = 0; col < width; ++col)
                    {
                        var iterVal = iterationEstimates[i][col, row];
                        iterationMoments[i][col, row] += iterVal * iterVal / curIteration;

                        var value = pixelValues[i][col, row];
                        var variance = iterationMoments[i][col, row] - value * value;
                        var moment = moments[i][col, row];


                        if (variance > 0 && moment > 0)
                            varianceFactors[i][col, row] = moment / variance;
                        else
                            varianceFactors[i][col, row] = 1;
                    }
                });
            }
        }

        public void Add(int cameraPathEdges, int lightPathEdges, int totalEdges,
                        Pixel filmPoint, RgbColor value)
        {
            if (!Enablefactor) return;
            if (isReady) return;
            bool isLightTracer = cameraPathEdges == 0;
            //if (isLightTracer && totalEdges > 1)
            if (isLightTracer && totalEdges == 2)
            {
                moments[totalEdges - 2].AtomicAdd(filmPoint.Col, filmPoint.Row,
                    (value * value / curIteration).Average);
                pixelValues[totalEdges - 2].AtomicAdd(filmPoint.Col, filmPoint.Row,
                    value.Average / curIteration);
                iterationEstimates[totalEdges - 2].AtomicAdd(filmPoint.Col, filmPoint.Row, value.Average);
            }
        }

        public float Get(int cameraPathEdges, int totalEdges, Pixel filmPoint)
        {
            if (!Enablefactor) return 1.0f;
            if (!isReady) return 1.0f;
            if (totalEdges != 2) return 1.0f;
            return varianceFactors[totalEdges - 2].GetPixel(filmPoint.Col, filmPoint.Row);
        }

        public void WriteToFiles(string basename)
        {
            if (!Enablefactor) return;
            if (!isReady) return;
            for (int i = 0; i < varianceFactors.Count; ++i)
            {
                var filename = Path.Join(basename, $"variance-factors-light-tracer-{i + 2}.exr");
                if (UseFilteredFactors) // write filted and accurate factors separately so that we dont override by accident
                    filename = Path.Join(basename, $"variance-factors-light-tracer-{i + 2}-filtered.exr");
                varianceFactors[i].WriteToFile(filename);

            }
        }

        bool isReady = false;
        bool UseFilteredFactors = false;
        int numTrainSamples = 0;
        int curIteration = 0;
        bool Enablefactor = false;
        protected List<MonochromeImage> iterationEstimates;
        protected List<MonochromeImage> iterationMoments;
        protected List<MonochromeImage> moments;
        protected List<MonochromeImage> pixelValues;
        protected List<MonochromeImage> varianceFactors;
    }

    public class VarAwareNextEvtFactors
    {
        public VarAwareNextEvtFactors(int maxDepth, int width, int height, bool UseFiltered, int numTrainSamples, bool enablefactor)
        {
            this.Enablefactor = enablefactor;
            if (!enablefactor) return;
            moments = new(maxDepth - 1);
            pixelValues = new(maxDepth - 1);
            varianceFactors = new(maxDepth - 1);
            iterationEstimates = new(maxDepth - 1);
            iterationMoments = new(maxDepth - 1);
            for (int i = 1; i < maxDepth; ++i)
            { // all depths with correlated next event (i.e., no DI)
                moments.Add(new(width, height));
                pixelValues.Add(new(width, height));
                varianceFactors.Add(new(width, height));
                iterationEstimates.Add(new(width, height));
                iterationMoments.Add(new(width, height));


            }

            this.numTrainSamples = numTrainSamples;
            this.UseFilteredFactors = UseFiltered;
        }

        public VarAwareNextEvtFactors(int maxDepth, string basename, bool UseFiltered, bool enablefactor)
        {
            this.Enablefactor = enablefactor;
            if (!enablefactor) return;
            varianceFactors = new(maxDepth - 1);
            for (int i = 2; i <= maxDepth; ++i)
            { // all depths with correlated next event (i.e., no DI)

                var filename = Path.Join(basename, $"variance-factors-nextevt-{i}.exr");
                if (UseFiltered)
                    filename = Path.Join(basename, $"variance-factors-nextevt-{i}-filtered.exr");
                varianceFactors.Add(new MonochromeImage(filename));
                isReady = true;
            }
        }

        public void StartIteration()
        {
            if (!Enablefactor) return;

            curIteration++;
            if (isReady) return;
            // Scale values of the previous iteration to account for having more samples
            if (curIteration > 1)
            {
                for (int i = 0; i < moments.Count; ++i)
                {
                    pixelValues[i].Scale((curIteration - 1.0f) / curIteration);

                    moments[i].Scale((curIteration - 1.0f) / curIteration);
                    iterationMoments[i].Scale((curIteration - 1.0f) / curIteration);

                    iterationEstimates[i].Scale(0);
                }
            }
        }

        public void EndIteration(uint iteration, Scene scene)
        {
            if (!Enablefactor) return;
            if (isReady) return;

            if (UseFilteredFactors)
                ComputeFilteredFactors();
            else
                ComputeAccurateFactors();

            if (iteration == numTrainSamples - 1)
            {
                isReady = true;
            }
        }

        void ComputeFilteredFactors()
        {
            int width = moments[0].Width;
            int height = moments[0].Height;
            MonochromeImage momentBuffer = new(width, height);
            MonochromeImage varianceBuffer = new(width, height);
            int blurRadius = 4;

            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < moments.Count; ++i)
            {
                {
                    // Estimate the pixel variances:
                    // First, we blur the image. Then, we subtract the blurred version from the original.
                    // Finally, we compute and square the difference, multiplying by the number of iterations
                    // to obtain a coarse estimate of the variance in a single iteration.
                    SimpleImageIO.Filter.RepeatedBox(pixelValues[i], varianceBuffer, blurRadius);
                    Parallel.For(0, height, row => {
                        for (int col = 0; col < width; ++col)
                        {
                            var value = pixelValues[i].GetPixel(col, row);
                            var delta = value - varianceBuffer.GetPixel(col, row);
                            var variance = delta * delta * curIteration;
                            varianceBuffer.SetPixel(col, row, variance);
                        }
                    });
                    SimpleImageIO.Filter.RepeatedBox(varianceBuffer, varianceFactors[i], blurRadius);

                    // Also filter the second moment estimates
                    SimpleImageIO.Filter.RepeatedBox(moments[i], momentBuffer, blurRadius);

                    // Compute the ratio for all non-zero pixels
                    Parallel.For(0, height, row => {
                        for (int col = 0; col < width; ++col)
                        {
                            var variance = varianceFactors[i].GetPixel(col, row);
                            var moment = momentBuffer.GetPixel(col, row);
                            if (variance > 0 && moment > 0)
                            {
                                varianceBuffer.SetPixel(col, row, moment / variance);
                            }
                            else
                            {
                                varianceBuffer.SetPixel(col, row, 1);
                            }
                        }
                    });

                    // Apply a wide filter to the ratio image, too
                    SimpleImageIO.Filter.RepeatedBox(varianceBuffer, varianceFactors[i], blurRadius);
                }
            }
        }

        void ComputeAccurateFactors()
        {
            int width = moments[0].Width;
            int height = moments[0].Height;

            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < moments.Count; ++i)
            {
                Parallel.For(0, height, row =>
                {
                    for (int col = 0; col < width; ++col)
                    {
                        var iterVal = iterationEstimates[i][col, row];
                        iterationMoments[i][col, row] += iterVal * iterVal / curIteration;

                        var value = pixelValues[i][col, row];
                        var variance = iterationMoments[i][col, row] - value * value;
                        var moment = moments[i][col, row];


                        if (variance > 0 && moment > 0)
                            varianceFactors[i][col, row] = moment / variance;
                        else
                            varianceFactors[i][col, row] = 1;
                    }
                });
            }
        }

        public void Add(int cameraPathEdges, int lightPathEdges, int totalEdges,
                        Pixel filmPoint, RgbColor value)
        {
            if (!Enablefactor) return;
            if (isReady) return;
            bool isNextEvent = lightPathEdges == 0 && cameraPathEdges == totalEdges - 1;
            //if (isNextEvent && totalEdges >=2)
            if (isNextEvent && totalEdges ==2)
            {
                moments[totalEdges - 2].AtomicAdd(filmPoint.Col, filmPoint.Row,
                    (value * value / curIteration).Average);
                pixelValues[totalEdges - 2].AtomicAdd(filmPoint.Col, filmPoint.Row,
                    value.Average / curIteration);
                iterationEstimates[totalEdges - 2].AtomicAdd(filmPoint.Col, filmPoint.Row, value.Average);
            }
        }

        public float Get(int cameraPathEdges, int totalEdges, Pixel filmPoint)
        {
            if (!Enablefactor) return 1.0f;
            if (!isReady) return 1.0f;
            if (totalEdges != 2) return 1.0f;
            return varianceFactors[totalEdges - 2].GetPixel(filmPoint.Col, filmPoint.Row);
        }

        public void WriteToFiles(string basename)
        {
            if (!Enablefactor) return;
            if (!isReady) return;
            for (int i = 0; i < varianceFactors.Count; ++i)
            {
                var filename = Path.Join(basename, $"variance-factors-nextevt-{i + 2}.exr");
                if (UseFilteredFactors) // write filted and accurate factors separately so that we dont override by accident
                    filename = Path.Join(basename, $"variance-factors-nextevt-{i + 2}-filtered.exr");
                varianceFactors[i].WriteToFile(filename);

            }
        }

        bool isReady = false;
        bool UseFilteredFactors = false;
        int numTrainSamples = 0;
        int curIteration = 0;
        bool Enablefactor = false;
        protected List<MonochromeImage> iterationEstimates;
        protected List<MonochromeImage> iterationMoments;
        protected List<MonochromeImage> moments;
        protected List<MonochromeImage> pixelValues;
        protected List<MonochromeImage> varianceFactors;
    }

    protected override void RegisterSample(RgbColor weight, float misWeight, Pixel pixel,
                                           int cameraPathLength, int lightPathLength, int fullLength) {
        base.RegisterSample(weight, misWeight, pixel, cameraPathLength, lightPathLength, fullLength);
        mergeFactors.Add(cameraPathLength, lightPathLength, fullLength, pixel, weight);
        lightTracerFactors.Add(cameraPathLength, lightPathLength, fullLength, pixel, weight);
        nextevtFactors.Add(cameraPathLength, lightPathLength, fullLength, pixel, weight);
        hitFactors.Add(cameraPathLength, lightPathLength, fullLength, pixel, weight);
    }

    protected override void OnEndIteration(uint iteration) {
        base.OnEndIteration(iteration);

        mergeFactors.EndIteration(iteration, Scene);
        lightTracerFactors.EndIteration(iteration, Scene);
        nextevtFactors.EndIteration(iteration, Scene);
        hitFactors.EndIteration(iteration, Scene);
    }

    protected override void OnStartIteration(uint iteration) {
        base.OnStartIteration(iteration);

        mergeFactors.StartIteration();
        lightTracerFactors.StartIteration();
        nextevtFactors.StartIteration();
        hitFactors.StartIteration();
    }

    public override void Render(Scene scene)
    {
        string path = Path.Join(scene.FrameBuffer.Basename, "variance-factors");
        InitFactors(scene, path);

        base.Render(scene);

        // We dont want to write to files if we define precomputed files
        if (!ReadFromPrecompute)
        {
            mergeFactors.WriteToFiles(path);
            lightTracerFactors.WriteToFiles(path);
            nextevtFactors.WriteToFiles(path);
            hitFactors.WriteToFiles(path);
        }
            
    }

    protected virtual void InitFactors(Scene scene, string path)
    {
        if (ReadFromPrecompute)
        {
            mergeFactors = new VarAwareMergingFactors(MaxDepth, path, UseFilteredFactors);
            lightTracerFactors = new VarAwareLightTracerFactors(MaxDepth, path, UseFilteredFactors, EnableLightTracerFactor);
            nextevtFactors = new VarAwareNextEvtFactors(MaxDepth, path, UseFilteredFactors, EnableNextEvtFactor);
            hitFactors = new VarAwareHitFactors(MaxDepth, path, UseFilteredFactors, EnableHitFactor);
        }
        else
        {
            mergeFactors = new VarAwareMergingFactors(MaxDepth, scene.FrameBuffer.Width, scene.FrameBuffer.Height, UseFilteredFactors, NumTrainSamples);
            lightTracerFactors = new VarAwareLightTracerFactors(MaxDepth, scene.FrameBuffer.Width, scene.FrameBuffer.Height, UseFilteredFactors, NumTrainSamples, EnableLightTracerFactor);
            nextevtFactors = new VarAwareNextEvtFactors(MaxDepth, scene.FrameBuffer.Width, scene.FrameBuffer.Height, UseFilteredFactors, NumTrainSamples, EnableNextEvtFactor);
            hitFactors = new VarAwareHitFactors(MaxDepth, scene.FrameBuffer.Width, scene.FrameBuffer.Height, UseFilteredFactors, NumTrainSamples, EnableHitFactor);
        }
    }

    public override float LightTracerMis(PathVertex lightVertex, in BidirPathPdfs pathPdfs, Pixel pixel, float distToCam)
    {
        var correlRatio = new CorrelAwareRatios(pathPdfs, distToCam, lightVertex.FromBackground);

        float footprintRadius = float.Sqrt(1.0f / pathPdfs.PdfsCameraToLight[0]);

        float radius = ComputeLocalMergeRadius(footprintRadius);
        float sumReciprocals = LightPathReciprocals(-1, pathPdfs, pixel, radius, correlRatio);
        sumReciprocals /= NumLightPaths.Value;
        sumReciprocals /= lightTracerFactors.Get(0, pathPdfs.NumPdfs, pixel);
        sumReciprocals += 1;

        return 1 / sumReciprocals;
    }

    public override float MergeMis(in CameraPath cameraPath, in PathVertex lightVertex, in BidirPathPdfs pathPdfs)
    {
        int numPdfs = cameraPath.Vertices.Count + lightVertex.Depth;
        // Compute the acceptance probability approximation
        int lastCameraVertexIdx = cameraPath.Vertices.Count - 1;
        float radius = ComputeLocalMergeRadius(cameraPath.FootprintRadius);    
        float mergeApproximation = pathPdfs.PdfsLightToCamera[lastCameraVertexIdx]
                                 * MathF.PI * radius * radius * NumLightPaths.Value;

        var correlRatio = new CorrelAwareRatios(pathPdfs, cameraPath.Distances[0], lightVertex.FromBackground);
        if (!DisableCorrelAwareMIS) mergeApproximation *= correlRatio[lastCameraVertexIdx];

        if (mergeApproximation == 0.0f) return 0.0f;
        // Compute the variance factor for this merge
        mergeApproximation *= mergeFactors.Get(cameraPath.Vertices.Count, numPdfs, cameraPath.Pixel);

        // Compute reciprocals for hypothetical connections along the camera sub-path
        float sumReciprocals = 0.0f;
        sumReciprocals +=
            CameraPathReciprocals(lastCameraVertexIdx, pathPdfs, cameraPath.Pixel, radius, correlRatio)
            / MathF.Pow(mergeApproximation, 1.0f);
        sumReciprocals +=
            LightPathReciprocals(lastCameraVertexIdx, pathPdfs, cameraPath.Pixel, radius, correlRatio)
            / MathF.Pow(mergeApproximation, 1.0f);

        // Add the reciprocal for the connection that replaces the last light path edge
        if (lightVertex.Depth > 1 && NumConnections > 0)
            sumReciprocals += BidirSelectDensity(cameraPath.Pixel) / MathF.Pow(mergeApproximation, 1.0f);

        return 1 / sumReciprocals;
    }

    public override float NextEventMis(in CameraPath cameraPath, in BidirPathPdfs pathPdfs, bool isBackground)
    {
        var correlRatio = new CorrelAwareRatios(pathPdfs, cameraPath.Distances[0], isBackground);

        float sumReciprocals = 1.0f;

        var factor = nextevtFactors.Get(cameraPath.Vertices.Count, cameraPath.Vertices.Count + 1, cameraPath.Pixel);
        var pdfNextEvent = pathPdfs.PdfNextEvent * factor;

        // Hitting the light source
        if (EnableHitting)
        {
            var hitFactor = hitFactors.Get(cameraPath.Vertices.Count, pathPdfs.NumPdfs, cameraPath.Pixel);
            sumReciprocals += pathPdfs.PdfsCameraToLight[^1] * hitFactor / pdfNextEvent;
        }

        // All bidirectional connections
        float radius = ComputeLocalMergeRadius(cameraPath.FootprintRadius);
        sumReciprocals +=
            CameraPathReciprocals(cameraPath.Vertices.Count - 1, pathPdfs, cameraPath.Pixel, radius, correlRatio)
            / pdfNextEvent;

        return 1 / sumReciprocals;
    }

    public override float EmitterHitMis(in CameraPath cameraPath, in BidirPathPdfs pathPdfs)
    {
        var correlRatio = new CorrelAwareRatios(pathPdfs, cameraPath.Distances[0], false);
        float sumReciprocals = 1.0f;

        var hitFactor = hitFactors.Get(cameraPath.Vertices.Count, pathPdfs.NumPdfs, cameraPath.Pixel);
        float pdfThis = pathPdfs.PdfsCameraToLight[^1] * hitFactor;

        // Next event estimation
        var neeFactor = nextevtFactors.Get(cameraPath.Vertices.Count - 1, cameraPath.Vertices.Count, cameraPath.Pixel);
        sumReciprocals += pathPdfs.PdfNextEvent * neeFactor / pdfThis;

        // All connections along the camera path
        float radius = ComputeLocalMergeRadius(cameraPath.FootprintRadius);
        sumReciprocals +=
            CameraPathReciprocals(cameraPath.Vertices.Count - 2, pathPdfs, cameraPath.Pixel, radius, correlRatio)
            / pdfThis;

        return 1 / sumReciprocals;
    }

    protected override float CameraPathReciprocals(int lastCameraVertexIdx, in BidirPathPdfs pdfs,
                                                  Pixel pixel, float radius, in CorrelAwareRatios correlRatio)
    {
        float sumReciprocals = 0.0f;
        float nextReciprocal = 1.0f;

        for (int i = lastCameraVertexIdx; i > 0; --i) {
            // Merging at this vertex
            if (EnableMerging) {
                float acceptProb = pdfs.PdfsLightToCamera[i] * MathF.PI * radius * radius;
                if (!DisableCorrelAwareMIS) acceptProb *= correlRatio[i];
                acceptProb *= mergeFactors.Get(i + 1, pdfs.NumPdfs, pixel);
                sumReciprocals += nextReciprocal * NumLightPaths.Value * acceptProb;
            }

            nextReciprocal *= pdfs.PdfsLightToCamera[i] / pdfs.PdfsCameraToLight[i];

            // Connecting this vertex to the next one along the camera path
            if (NumConnections > 0) sumReciprocals += nextReciprocal * BidirSelectDensity(pixel);
        }

        // Light tracer
        if (EnableLightTracer)
        {
            var factor = lightTracerFactors.Get(0, pdfs.NumPdfs, pixel);
            sumReciprocals +=
                nextReciprocal * pdfs.PdfsLightToCamera[0] / pdfs.PdfsCameraToLight[0] * NumLightPaths.Value * factor;
        }


        // Merging directly visible (almost the same as the light tracer!)
        if (MergePrimary)
        {
            sumReciprocals +=nextReciprocal * NumLightPaths.Value * pdfs.PdfsLightToCamera[0]
                * MathF.PI * radius * radius * mergeFactors.Get(1, pdfs.NumPdfs, pixel);
        }


        return sumReciprocals;
    }

    protected override float LightPathReciprocals(int lastCameraVertexIdx, in BidirPathPdfs pdfs,
                                                 Pixel pixel, float radius, in CorrelAwareRatios correlRatio)
    {
        float sumReciprocals = 0.0f;
        float nextReciprocal = 1.0f;

        for (int i = lastCameraVertexIdx + 1; i < pdfs.NumPdfs; ++i)
        {
            if (i == pdfs.NumPdfs - 1) // Next event
            {
                var factor = nextevtFactors.Get(i, pdfs.NumPdfs, pixel);
                var nextEvtPdf = pdfs.PdfNextEvent * factor;
                sumReciprocals += nextReciprocal * nextEvtPdf / pdfs.PdfsLightToCamera[i];
            }

            if (i < pdfs.NumPdfs - 1 && (MergePrimary || i > 0))
            { // no merging on the emitter itself
                // Account for merging at this vertex
                if (EnableMerging)
                {
                    float acceptProb = pdfs.PdfsCameraToLight[i] * MathF.PI * radius * radius;
                    if (!DisableCorrelAwareMIS) acceptProb *= correlRatio[i];
                    acceptProb *= mergeFactors.Get(i + 1, pdfs.NumPdfs, pixel);
                    sumReciprocals += nextReciprocal * NumLightPaths.Value * acceptProb;
                }
            }
            nextReciprocal *= pdfs.PdfsCameraToLight[i] / pdfs.PdfsLightToCamera[i];

            // Account for connections from this vertex to its ancestor
            if (i < pdfs.NumPdfs - 2) // Connections to the emitter (next event) are treated separately
                if (NumConnections > 0) sumReciprocals += nextReciprocal * BidirSelectDensity(pixel);
        }

        if (EnableHitting)
        {
            var hitfactor = hitFactors.Get(pdfs.NumPdfs, pdfs.NumPdfs, pixel);
            sumReciprocals += nextReciprocal * hitfactor; // Hitting the emitter directly
        }

        return sumReciprocals;
    }
}