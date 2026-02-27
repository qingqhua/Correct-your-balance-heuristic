
namespace VarAwareVCM;

public class Ours : VertexConnectionAndMerging
{
    public float[] CandidatesMerge;
    public float[] CandidatesLightTracer;
    public int NumTrainSamples = 0;
    public bool EnableLightTracerFactor = false;
    public int BlurRadius = 8;

    protected CorrectionFactorsBase mergeFactors;
    bool UseFilteredFactors = true;

    public class OurFactors : CorrectionFactorsBase
    {
        public OurFactors(int maxDepth, int width, int height, bool useFilteredFactors, 
            int numTrainSamples, bool enablelightTracerFactor, float[] candsMerge, float[] candsLightTracer, int blurRadius)
        {
            this.enableLightTracerFactor = enablelightTracerFactor;
            // switch between accurate and filtered factors. The later only need 1 iter to estimate.
            this.UseFilteredFactors = useFilteredFactors;
            this.numTrainSamples = numTrainSamples;
            this.blurRadius = blurRadius;

            // Initialize merging factor
            {
                mergingFactor = new(width, height);
                mergingVariance = new(width, height);
                mergingFactor.Fill(1.0f);
                mergingVariance.Fill(float.MaxValue);

                int numCandidates = candsMerge.Count();
                candsMerge.ToList().ForEach(i => Console.Write("{0}\t", i));
                Console.WriteLine();

                iterationMomentsMerge = new(numCandidates);
                iterationEstimateMerge = new(numCandidates);
                mergingFactorCands = new(numCandidates);
                pixelValuesMerge = new(numCandidates);
                for (int k = 0; k < numCandidates; k++)
                {
                    iterationMomentsMerge.Add(new(width, height));
                    iterationEstimateMerge.Add(new(width, height));
                    mergingFactorCands.Add(new(width, height));
                    pixelValuesMerge.Add(new(width, height));

                    mergingFactorCands[^1].Fill(candsMerge[k]);
                }
            }

            if (!enableLightTracerFactor) return;
            {
                lightTracerFactor = new(width, height);
                lightTracerVariance = new(width, height);
                lightTracerFactor.Fill(1.0f);
                lightTracerVariance.Fill(float.MaxValue);

                int numCandidates = candsLightTracer.Count();
                candsLightTracer.ToList().ForEach(i => Console.Write("{0}\t", i));
                Console.WriteLine();

                iterationMomentsLightTracer = new(numCandidates);
                iterationEstimateLightTracer = new(numCandidates);
                pixelValuesLightTracer = new(numCandidates);
                lightTracerFactorCands = new(numCandidates);

                for (int k = 0; k < numCandidates; k++)
                {
                    iterationMomentsLightTracer.Add(new(width, height));
                    iterationEstimateLightTracer.Add(new(width, height));
                    pixelValuesLightTracer.Add(new(width, height));
                    lightTracerFactorCands.Add(new(width, height));

                    lightTracerFactorCands[^1].Fill(candsLightTracer[k]);
                }
            }
        }

        public override void StartIteration()
        {
            curIteration++;

            if (isReady) return;

            // Scale values of the previous iteration to account for having more samples
            if (curIteration > 1)
            {
                for (int k = 0; k < mergingFactorCands.Count; k++)
                {
                    iterationMomentsMerge[k].Scale((curIteration - 1.0f) / curIteration);
                    pixelValuesMerge[k].Scale((curIteration - 1.0f) / curIteration);

                    iterationEstimateMerge[k].Scale(0);
                }

                if (!enableLightTracerFactor) return;
                for (int k = 0; k < lightTracerFactorCands.Count; k++)
                {
                    iterationEstimateLightTracer[k].Scale((curIteration - 1.0f) / curIteration);
                    pixelValuesLightTracer[k].Scale((curIteration - 1.0f) / curIteration);

                    iterationEstimateLightTracer[k].Scale(0);
                }
            }
        }

        public override void EndIteration(uint iteration, Scene scene)
        {
            if (isReady) return;
            if (iteration == numTrainSamples - 1)
            {
                if (UseFilteredFactors) // Note if filtered sample is more than 1, moments update needs to be called in every iteration
                    ComputeFilteredFactors(scene);
            }
        }

        void MomentsUpdate()
        {
            for (int c = 0; c < mergingFactorCands.Count; c++)
            {
                var iterVal = iterationEstimateMerge[c];
                iterationMomentsMerge[c] += iterVal * iterVal / curIteration;

            }

            if (!enableLightTracerFactor) return;

            for (int c = 0; c < lightTracerFactorCands.Count; c++)
            {
                var iterVal = iterationEstimateLightTracer[c];
                iterationMomentsLightTracer[c] += iterVal * iterVal / curIteration;
            }
        }

        public override void Add(int cameraPathEdges, int lightPathEdges, int totalEdges,
                        Pixel pixel, RgbColor value, float kernelWeight, float actualMisweight, float mergeSumMis, float lightTracerMis, float nextEvtMis)
        {
            if (isReady) return;
            AddMergeEstimate(cameraPathEdges, lightPathEdges, totalEdges, pixel, value, kernelWeight, actualMisweight, mergeSumMis);
            AddLighTracerEstimate(cameraPathEdges, totalEdges, pixel, value, actualMisweight, lightTracerMis);
        }

        private void AddMergeEstimate(int cameraPathEdges, int lightPathEdges, int totalEdges,
                Pixel pixel, RgbColor value, float kernelWeight, float actualMisweight, float mergeSumMis)
        {
            bool isMerge = lightPathEdges > 0 && lightPathEdges + cameraPathEdges == totalEdges;

            if (totalEdges < 3) return;
            for (int k = 0; k < mergingFactorCands.Count; ++k) // loop over candidates
            {
                var w = 0.0f;
                var c = mergingFactorCands[k].GetPixel(pixel.Col, pixel.Row);
                var denom = mergeSumMis * c + (1.0f - mergeSumMis);
                if (isMerge && cameraPathEdges > 1) // Primary merges have zero covariance
                {
                    var nom = c * actualMisweight;
                    w = nom / denom;
                }
                else if (!isMerge)// in other techniques
                {
                    var nom = actualMisweight;
                    w = nom / denom;
                }

                w *= kernelWeight; //TODO: check if we are multiplying twice on second moment estimate

                iterationEstimateMerge[k].AtomicAdd(pixel.Col, pixel.Row,
                    w * value.Average);
                pixelValuesMerge[k].AtomicAdd(pixel.Col, pixel.Row,
                        w * value.Average / curIteration);
            }
        }

        private void AddLighTracerEstimate(int cameraPathEdges, int totalEdges,
                        Pixel pixel, RgbColor value, float actualMisweight, float lightTracerMis)
        {
            if (!enableLightTracerFactor) return;
            if (totalEdges != 2) return; // Light tracer factor only controls direct illumination
            //if (totalEdges < 2) return; // Light tracer factor only controls direct illumination
            bool isLightTracer = cameraPathEdges == 0;
            for (int k = 0; k < lightTracerFactorCands.Count; ++k) // loop over candidates
            {
                var w = 0.0f;
                var c = lightTracerFactorCands[k].GetPixel(pixel.Col, pixel.Row);
                var denom = lightTracerMis * c + (1.0f - lightTracerMis);
                if (isLightTracer)
                {
                    var nom = c * actualMisweight;
                    w = nom / denom;
                }
                else if (!isLightTracer)// in other techniques
                {
                    var nom = actualMisweight;
                    w = nom / denom;
                }
                iterationEstimateLightTracer[k].AtomicAdd(pixel.Col, pixel.Row,
                    w * value.Average);
                pixelValuesLightTracer[k].AtomicAdd(pixel.Col, pixel.Row,
                        w * value.Average / curIteration);
            }
        }

        public override float Get(int cameraPathEdges, int totalEdges, Pixel pixel)
        {
            if (!isReady) return 1.0f;
            bool isLightTracer = cameraPathEdges == 0;
            if (isLightTracer)
            {
                if (!enableLightTracerFactor) return 1.0f;
                if (totalEdges != 2) return 1.0f; // light tracer only controls direct illumination
                //if (totalEdges < 2) return 1.0f; 
                return lightTracerFactor.GetPixel(pixel.Col, pixel.Row);
            }

            // Merge factors is a global for all indirect illumination
            if (cameraPathEdges < 2) return 1.0f;
            if (totalEdges < 3) return 1.0f;
            return mergingFactor
                .GetPixel(pixel.Col, pixel.Row);
        }

        public override void WriteFactors(string basename, int mindepth, int maxdepth)
        {
            if (!isReady) return;
            var prefix = Path.Join($"{basename}", "variance-factors");

            var filename = $"{prefix}-merge.exr";
            if (UseFilteredFactors)
                filename = $"{prefix}-merge-filtered.exr";
            var rgb = new RgbImage(mergingFactor);
            rgb.WriteToFile(filename);

            if (!enableLightTracerFactor) return;
            filename = $"{prefix}-light-tracer.exr";
            if (UseFilteredFactors)
                filename = $"{prefix}-light-tracer-filtered.exr";
            lightTracerFactor.WriteToFile(filename);
        }

        public override void WriteMoments(string basename)
        {
            if (!isReady) return;

            var prefix = Path.Join($"{basename}", "Misc", "Merge");
            for (int i = 0; i < mergingFactorCands.Count; i++)
            {
                iterationMomentsMerge[i].WriteToFile(Path.Join(prefix, $"iterationMoments-{mergingFactorCands[i][0, 0]}.exr"));
                pixelValuesMerge[i].WriteToFile(Path.Join(prefix, $"pixel-{mergingFactorCands[i][0, 0]}.exr"));

            }
            if (!enableLightTracerFactor) return;
            prefix = Path.Join($"{basename}", "Misc", "LightTracer");
            for (int i = 0; i < lightTracerFactorCands.Count; i++)
            {
                iterationMomentsLightTracer[i].WriteToFile(Path.Join(prefix, $"iterationMoments-{lightTracerFactorCands[i][0, 0]}.exr"));
                pixelValuesLightTracer[i].WriteToFile(Path.Join(prefix, $"pixel-{lightTracerFactorCands[i][0, 0]}.exr"));
            }
        }

        void ComputeFilteredFactors(Scene scene)
        {
            ComputeFilteredLightTracerFactor(scene);
            ComputeFilteredMergeFactor(scene);
            isReady = true;
        }

        void ComputeFilteredMergeFactor(Scene scene)
        {
            int width = iterationMomentsMerge[0].Width;
            int height = iterationMomentsMerge[0].Height;
            MonochromeImage pixelBuffer = new(width, height);
            MonochromeImage varianceBuffer = new(width, height);
            for (int c = 0; c < mergingFactorCands.Count; ++c) // candidates
            {
                // Moments update
                var iterVal = iterationEstimateMerge[c];
                iterationMomentsMerge[c] += iterVal * iterVal / curIteration;

                // Blur the  moments and pixel separately
                SimpleImageIO.Filter.RepeatedBox(iterationMomentsMerge[c], varianceBuffer, blurRadius);
                SimpleImageIO.Filter.RepeatedBox(pixelValuesMerge[c], pixelBuffer, blurRadius);

                Parallel.For(0, height, row =>
                {
                    for (int col = 0; col < width; ++col)
                    {
                        if (scene.FrameBuffer.Image[col, row] == RgbColor.Black)
                            continue;

                        {
                            var pixelSqr = pixelBuffer[col, row] * pixelBuffer[col, row];
                            var variance = varianceBuffer[col, row] / pixelSqr;
                            var diff = variance - this.mergingVariance[col, row];
                            // Compare with the blurred variance and get the minimum
                            if (diff < 0 && variance > 0)
                            {
                                this.mergingVariance[col, row] = variance;
                                mergingFactor[col, row] = mergingFactorCands[c].GetPixel(col, row);
                            }
                        }
                    }
                });
            }

            SimpleImageIO.Filter.RepeatedBox(mergingFactor, mergingFactor, blurRadius);
        }

        void ComputeFilteredLightTracerFactor(Scene scene)
        {
            if (!enableLightTracerFactor) return;
            int width = iterationMomentsMerge[0].Width;
            int height = iterationMomentsMerge[0].Height;

            MonochromeImage varianceBuffer = new(width, height);
            MonochromeImage pixelBuffer = new(width, height);

            for (int c = 0; c < lightTracerFactorCands.Count; ++c) // candidates
            {
                // Moments update
                var iterVal = iterationEstimateLightTracer[c];
                iterationMomentsLightTracer[c] += iterVal * iterVal / curIteration;

                SimpleImageIO.Filter.RepeatedBox(iterationMomentsLightTracer[c], varianceBuffer, blurRadius);
                SimpleImageIO.Filter.RepeatedBox(pixelValuesLightTracer[c], pixelBuffer, blurRadius);

                Parallel.For(0, height, row =>
                {
                    for (int col = 0; col < width; ++col)
                    {
                        if (scene.FrameBuffer.Image[col, row] == RgbColor.Black)
                            continue;

                        var pixelSqr = pixelBuffer[col, row] * pixelBuffer[col, row];
                        var variance = (varianceBuffer[col, row] - pixelSqr) / pixelSqr;

                        // Compare with the blurred variance and get the minimum
                        if (variance < this.lightTracerVariance[col, row] && variance > 0)
                        {
                            this.lightTracerVariance[col, row] = variance;
                            lightTracerFactor[col, row] = lightTracerFactorCands[c].GetPixel(col, row);
                        }
                    }
                });
            }

            SimpleImageIO.Filter.RepeatedBox(lightTracerFactor, lightTracerFactor, blurRadius);
        }

        public override bool IsReady()
        {
            return isReady;
        }

        public bool isReady = false;
        public bool enableLightTracerFactor = false;
        bool UseFilteredFactors = false;
        public int numTrainSamples = 0;
        int curIteration = 0;
        int blurRadius = 8;

        List<MonochromeImage> iterationMomentsMerge;
        List<MonochromeImage> pixelValuesMerge;
        List<MonochromeImage> iterationEstimateMerge;

        List<MonochromeImage> mergingFactorCands;
        MonochromeImage mergingFactor;
        MonochromeImage mergingVariance;

        List<MonochromeImage> iterationMomentsLightTracer;
        List<MonochromeImage> pixelValuesLightTracer;
        List<MonochromeImage> iterationEstimateLightTracer;


        MonochromeImage lightTracerFactor;
        MonochromeImage lightTracerVariance;
        List<MonochromeImage> lightTracerFactorCands;
    }

    void OnMergeVarianceEstimate(RgbColor weight, float kernelWeight, float misWeight, Pixel pixel,
                            int cameraPathLength, int lightPathLength, int fullLength, float mergeMisSum, float lightTracerMis)
    {
        mergeFactors.Add(cameraPathLength, lightPathLength, fullLength, pixel, weight, kernelWeight, misWeight, mergeMisSum, lightTracerMis, 0);
    }


    protected override void OnEndIteration(uint iteration)
    {
        base.OnEndIteration(iteration);

        mergeFactors.EndIteration(iteration, Scene);
    }

    protected override void OnStartIteration(uint iteration)
    {
        base.OnStartIteration(iteration);

        mergeFactors.StartIteration();
    }

    public override void Render(Scene scene)
    {
        string path = Path.Join(scene.FrameBuffer.Basename);
        InitFactors(scene, path);

        base.Render(scene);
    }

    protected override void OnAfterRender()
    {
        base.OnAfterRender();
        string path = Path.Join(Scene.FrameBuffer.Basename);

        mergeFactors.WriteFactors(path, MinDepth, MaxDepth);
        mergeFactors.WriteMoments(path);
    }

    protected virtual void InitFactors(Scene scene, string path)
    {
        mergeFactors = new OurFactors(MaxDepth, scene.FrameBuffer.Width, scene.FrameBuffer.Height,
            UseFilteredFactors, NumTrainSamples, EnableLightTracerFactor, CandidatesMerge, CandidatesLightTracer, BlurRadius);
        
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
            / mergeApproximation;
        sumReciprocals +=
            LightPathReciprocals(lastCameraVertexIdx, pathPdfs, cameraPath.Pixel, radius, correlRatio)
            / mergeApproximation;

        // Add the reciprocal for the connection that replaces the last light path edge
        if (lightVertex.Depth > 1 && NumConnections > 0)
            sumReciprocals += BidirSelectDensity(cameraPath.Pixel) / mergeApproximation;

        return 1 / sumReciprocals;
    }

    protected override float CameraPathReciprocals(int lastCameraVertexIdx, in BidirPathPdfs pdfs,
                                          Pixel pixel, float radius, in CorrelAwareRatios correlRatio)
    {
        float sumReciprocals = 0.0f;
        float nextReciprocal = 1.0f;

        for (int i = lastCameraVertexIdx; i > 0; --i)
        {
            // Merging at this vertex
            if (EnableMerging)
            {
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
            sumReciprocals +=
                    nextReciprocal * pdfs.PdfsLightToCamera[0] / pdfs.PdfsCameraToLight[0] * NumLightPaths.Value * mergeFactors.Get(0, pdfs.NumPdfs, pixel);
        }

        // Merging directly visible (almost the same as the light tracer!)
        if (MergePrimary)
            sumReciprocals += nextReciprocal * NumLightPaths.Value * pdfs.PdfsLightToCamera[0]
                * MathF.PI * radius * radius;

        return sumReciprocals;
    }

    public override float LightTracerMis(PathVertex lightVertex, in BidirPathPdfs pathPdfs, Pixel pixel, float distToCam)
    {
        var correlRatio = new CorrelAwareRatios(pathPdfs, distToCam, lightVertex.FromBackground);

        float footprintRadius = float.Sqrt(1.0f / pathPdfs.PdfsCameraToLight[0]);

        float radius = ComputeLocalMergeRadius(footprintRadius);
        float sumReciprocals = LightPathReciprocals(-1, pathPdfs, pixel, radius, correlRatio);
        sumReciprocals /= NumLightPaths.Value;
        sumReciprocals /= mergeFactors.Get(0, pathPdfs.NumPdfs, pixel);
        sumReciprocals += 1;

        return 1 / sumReciprocals;
    }

    protected override float LightPathReciprocals(int lastCameraVertexIdx, in BidirPathPdfs pdfs,
                                                 Pixel pixel, float radius, in CorrelAwareRatios correlRatio)
    {
        float sumReciprocals = 0.0f;
        float nextReciprocal = 1.0f;

        for (int i = lastCameraVertexIdx + 1; i < pdfs.NumPdfs; ++i)
        {
            if (i == pdfs.NumPdfs - 1) // Next event
                sumReciprocals += nextReciprocal * pdfs.PdfNextEvent / pdfs.PdfsLightToCamera[i];

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
        // Next event and hitting the emitter directly
        if (EnableHitting) sumReciprocals += nextReciprocal; // Hitting the emitter directly
        return sumReciprocals;
    }

    /// <returns>
    /// Sum of MIS weights for all merges along the path when using the proxy strategy, and the same sum
    /// but with each weight multiplied by the correl-aware MIS correction factor
    /// </returns>
    protected override void OnLightTracerSample(RgbColor weight, float misWeight, Pixel pixel,
                                       PathVertex lightVertex, in BidirPathPdfs pathPdfs, float distToCam)
    {
        if (weight == RgbColor.Black) return;
        if (mergeFactors.IsReady()) return;
        var correlRatio = new CorrelAwareRatios(pathPdfs, distToCam, lightVertex.FromBackground);
        float footprintRadius = float.Sqrt(1.0f / pathPdfs.PdfsCameraToLight[0]);
        float radius = ComputeLocalMergeRadius(footprintRadius);

        (float misMergeSum, float misLightTracer) = ComputeProxyMis(pathPdfs, pixel, correlRatio, radius, lightVertex.Depth + 1);
        OnMergeVarianceEstimate(weight, 1, misWeight, pixel, 0, lightVertex.Depth, lightVertex.Depth + 1, misMergeSum, misLightTracer);
    }

    protected override void OnNextEventSample(RgbColor weight, float misWeight, CameraPath cameraPath,
                                         float pdfNextEvent, float pdfHit, in BidirPathPdfs pathPdfs,
                                         Emitter emitter, Vector3 lightToSurface, SurfacePoint lightPoint)
    {
        if (weight == RgbColor.Black) return;
        if (mergeFactors.IsReady()) return;

        bool isBackground = false;
        if (emitter == null)
            isBackground = true;
        var correlRatio = new CorrelAwareRatios(pathPdfs, cameraPath.Distances[0], isBackground);
        float radius = ComputeLocalMergeRadius(cameraPath.FootprintRadius);

        (float misMergeSum, float misLightTracer) = ComputeProxyMis(pathPdfs, cameraPath.Pixel, correlRatio, radius, cameraPath.Vertices.Count + 1);
        OnMergeVarianceEstimate(weight, 1, misWeight, cameraPath.Pixel, cameraPath.Vertices.Count, 0, cameraPath.Vertices.Count + 1, misMergeSum, misLightTracer);
    }

    protected override void OnEmitterHitSample(RgbColor weight, float misWeight, CameraPath cameraPath,
                                      float pdfNextEvent, in BidirPathPdfs pathPdfs, Emitter emitter,
                                      Vector3 lightToSurface, SurfacePoint lightPoint)
    {
        if (weight == RgbColor.Black) return;
        if (mergeFactors.IsReady()) return;
        int lastCameraVertexIdx = cameraPath.Vertices.Count - 1;

        var correlRatio = new CorrelAwareRatios(pathPdfs, cameraPath.Distances[0], false);
        float radius = ComputeLocalMergeRadius(cameraPath.FootprintRadius);

        (float misMergeSum, float misLightTracer) = ComputeProxyMis(pathPdfs, cameraPath.Pixel, correlRatio, radius, cameraPath.Vertices.Count);
        OnMergeVarianceEstimate(weight, 1, misWeight, cameraPath.Pixel, cameraPath.Vertices.Count, 0, cameraPath.Vertices.Count, misMergeSum, misLightTracer);
    }

    protected override void OnBidirConnectSample(RgbColor weight, float misWeight, CameraPath cameraPath,
                                        PathVertex lightVertex, in BidirPathPdfs pathPdfs)
    {
        if (weight == RgbColor.Black) return;
        if (mergeFactors.IsReady()) return;
        int depth = lightVertex.Depth + cameraPath.Vertices.Count + 1;
        int lastCameraVertexIdx = cameraPath.Vertices.Count - 1;
        var correlRatio = new CorrelAwareRatios(pathPdfs, cameraPath.Distances[0], lightVertex.FromBackground);
        float radius = ComputeLocalMergeRadius(cameraPath.FootprintRadius);

        (float misMergeSum, float misLightTracer) = ComputeProxyMis(pathPdfs, cameraPath.Pixel, correlRatio, radius, depth);
        OnMergeVarianceEstimate(weight, 1, misWeight, cameraPath.Pixel, cameraPath.Vertices.Count, lightVertex.Depth, depth, misMergeSum, misLightTracer);
    }

    protected override void OnMergeSample(RgbColor weight, float kernelWeight, float misWeight,
                                     CameraPath cameraPath, PathVertex lightVertex, in BidirPathPdfs pathPdfs)
    {

        base.OnMergeSample(weight, kernelWeight, misWeight, cameraPath, lightVertex, pathPdfs);
        if (weight == RgbColor.Black) return;
        if (mergeFactors.IsReady()) return;

        var depth = cameraPath.Vertices.Count + lightVertex.Depth;
        var correlRatio = new CorrelAwareRatios(pathPdfs, cameraPath.Distances[0], lightVertex.FromBackground);
        float radius = ComputeLocalMergeRadius(cameraPath.FootprintRadius);

        (float misMergeSum, float misLightTracer) = ComputeProxyMis(pathPdfs, cameraPath.Pixel, correlRatio, radius, depth);
        OnMergeVarianceEstimate(weight, kernelWeight, misWeight, cameraPath.Pixel, cameraPath.Vertices.Count,
            lightVertex.Depth, depth, misMergeSum, misLightTracer);
    }


    private float ComputeLightTracerMISProxy(BidirPathPdfs pdfs, Pixel pixel,
                                  CorrelAwareRatios ratio, float radius)
    {
        // MIS weight of light tracing
        float ltRecip = LightPathReciprocalsForProxy(-1, pdfs, pixel, radius, ratio);
        float lt = 1 / (1 + ltRecip / NumLightPaths.Value);
        return lt;
    }

    private (float, float) ComputeProxyMis(BidirPathPdfs pdfs, Pixel pixel,
                                  CorrelAwareRatios ratio, float radius, int fulllength)
    {
        float misLight = 0, misMergeSum = 0;
        if (fulllength == 2 && EnableLightTracerFactor) // light tracer factor only applied on di
            misLight = ComputeLightTracerMISProxy(pdfs, pixel, ratio, radius);
        if (fulllength > 2) // primary merge should always be closed, apply merge factors on indirect
            misMergeSum = ComputeMergeSumProxy(pdfs, pixel, ratio, radius);

        return (misMergeSum, misLight);
    }

    private float ComputeMergeSumProxy(BidirPathPdfs pdfs, Pixel pixel,
                                      CorrelAwareRatios ratio, float radius)
    {
        // This loop over all merge techniques can be optimized by interleaving it with the other gathering
        // operations along the path.

        float misSum = 0;
        for (int i = 1; i < pdfs.NumPdfs - 1; ++i)
        {
            Debug.Assert(ratio[i] <= 1 + 1e-5f);

            float mergeApproximation =
                pdfs.PdfsLightToCamera[i] * MathF.PI * radius * radius * NumLightPaths.Value;

            if (!DisableCorrelAwareMIS) mergeApproximation *= ratio[i];
            if (mergeApproximation == 0) continue;
            float cam = CameraPathReciprocals(i, pdfs, pixel, radius, ratio);
            float lig = LightPathReciprocalsForProxy(i, pdfs, pixel, radius, ratio); // A hack to set all zero pdfs to 1

            Debug.Assert(!float.IsInfinity(lig) || !float.IsNaN(lig) || lig != 0);
            Debug.Assert(!float.IsInfinity(cam) || !float.IsNaN(cam) || cam != 0);

            float sumReciprocals = 0.0f;
            sumReciprocals += cam / mergeApproximation;
            sumReciprocals += lig / mergeApproximation;

            // Ratio between the merge and the equivalent connection technique
            int lightVertexDepth = pdfs.NumPdfs - i - 1;
            if (lightVertexDepth > 1 && NumConnections > 0) sumReciprocals += BidirSelectDensity(pixel) / mergeApproximation;
            Debug.Assert(sumReciprocals > 0);

            misSum += 1f / sumReciprocals;
        }

        return misSum;
    }

    protected float LightPathReciprocalsForProxy(int lastCameraVertexIdx, in BidirPathPdfs pdfs,
                                         Pixel pixel, float radius, in CorrelAwareRatios correlRatio)
    {
        float sumReciprocals = 0.0f;
        float nextReciprocal = 1.0f;

        for (int i = lastCameraVertexIdx + 1; i < pdfs.NumPdfs; ++i)
        {
            if (pdfs.PdfsLightToCamera[i] == 0)
                pdfs.PdfsLightToCamera[i] = 1; // TODO: Hack for preventing division zero

            if (i == pdfs.NumPdfs - 1) // Next event
                sumReciprocals += nextReciprocal * pdfs.PdfNextEvent / pdfs.PdfsLightToCamera[i];

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
        // Next event and hitting the emitter directly
        if (EnableHitting) sumReciprocals += nextReciprocal; // Hitting the emitter directly
        return sumReciprocals;
    }
}