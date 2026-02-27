namespace RIS;

/// <summary>
/// Implementation of the paper:
/// "Variance-aware multiple importance sampling" [Grittmann et al. 2019]
/// </summary>
public class VarAware : RISDI
{
    /// <summary>
    /// Use filtered the correction factor
    /// </summary>
    public bool UseFiltered = true;

    /// <summary>
    /// Determine how many samples we want to use to compute factor.
    /// </summary>
    public int NumTrainingSamples = 1;

    public CorrectionFactor correctionFactors;

    public class CorrectionFactor
    {
        public CorrectionFactor(int numTrainSamples, int width, int height, bool useFilteredFactors = true)
        {
            secondMoments = new(2);
            firstMoments = new(2);
            varianceFactors = new(2);
            for (int i = 0; i < 2; i++)
            {
                secondMoments.Add(new(width, height));
                firstMoments.Add(new(width, height));
                varianceFactors.Add(new(width, height));
                varianceFactors[i].Fill(1.0f);
            }

            variance = new(width, height); // # 1
            variance.Fill(float.MaxValue);

            // Switch between accurate and filtered factors. The later only need 1 iter to estimate.
            this.UseFilteredFactors = useFilteredFactors;
            this.numTrainSamples = numTrainSamples;
        }

        public void StartIteration()
        {
            curIteration++;

            if (isReady) return;

            // Scale values of the previous iteration to account for having more samples
            if (curIteration > 1)
            {
                for (int i = 0; i < 2; i++)
                {
                    firstMoments[i].Scale((curIteration - 1.0f) / curIteration);
                    secondMoments[i].Scale((curIteration - 1.0f) / curIteration);
                }
            }
        }

        void ComputeFiliteredFactors()
        {
            int width = secondMoments[0].Width;
            int height = secondMoments[0].Height;
            int blurRadius = 4;
            MonochromeImage momentBuffer = new(width, height);
            MonochromeImage varianceBuffer = new(width, height);

            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < secondMoments.Count; ++i)
            {
                // Estimate the pixel variances:
                // First, we blur the image. Then, we subtract the blurred version from the original.
                // Finally, we compute and square the difference, multiplying by the number of iterations
                // to obtain a coarse estimate of the variance in a single iteration.
                Filter.RepeatedBox(firstMoments[i], varianceBuffer, blurRadius);
                for (int row = 0; row < height; ++row)
                {
                    for (int col = 0; col < width; ++col)
                    {
                        var delta = firstMoments[i].GetPixel(col, row) - varianceBuffer.GetPixel(col, row);
                        var mean = varianceBuffer.GetPixel(col, row);

                        var variance = delta * delta * curIteration;
                        varianceBuffer.SetPixel(col, row, variance);
                    }
                }

                Filter.RepeatedBox(varianceBuffer, varianceFactors[i], blurRadius);

                // Also filter the second moment estimates
                Filter.RepeatedBox(secondMoments[i], momentBuffer, blurRadius);
                for (int row = 0; row < height; ++row)
                {
                    for (int col = 0; col < width; ++col)
                    {
                        var variance = varianceFactors[i].GetPixel(col, row);
                        var moment = momentBuffer.GetPixel(col, row);
                        if (variance > 0)
                        {
                            varianceBuffer.SetPixel(col, row, moment / variance);
                        }
                        else
                        {
                            varianceBuffer.SetPixel(col, row, 1);
                        }
                    }
                }

                // Apply a wide filter to the ratio image, too
                Filter.RepeatedBox(varianceBuffer, varianceFactors[i], blurRadius);
            }
        }

        void ComputeAccurateFactors()
        {
            int width = secondMoments[0].Width;
            int height = secondMoments[0].Height;


            // Compute the variance factors for use in the next iteration
            for (int k = 0; k < 2; ++k) // techniques
            {
                Parallel.For(0, height, row =>
                {
                    for (int col = 0; col < width; ++col)
                    {
                        var moment = secondMoments[k][col, row];
                        var firstMomentSqr = firstMoments[k][col, row] * firstMoments[k][col, row];
                        var variance = moment - firstMomentSqr; // variance for each candidates

                        // Use variance-aware MIS
                        if (variance > 0 && moment > 0)
                            varianceFactors[k].SetPixel(col, row, moment / variance);
                    }
                });
            }
        }

        public void EndIteration(uint iteration)
        {
            if (isReady) return;
            if (iteration == numTrainSamples - 1)
            {
                if (UseFilteredFactors)
                    ComputeFiliteredFactors();
                else
                    ComputeAccurateFactors();
                isReady = true;
            }

        }

        public void Add(int techIdx, Pixel pixel, RgbColor fVal)
        {
            if (isReady) return;

            float v = fVal.Average;

            secondMoments[techIdx].AtomicAdd(pixel.Col, pixel.Row, v * v / curIteration);
            firstMoments[techIdx].AtomicAdd(pixel.Col, pixel.Row, v / curIteration);
        }

        public float Get(Pixel pixel)
        {
            if (!isReady) return 1.0f;
            return varianceFactors[1].GetPixel(pixel.Col, pixel.Row) / varianceFactors[0].GetPixel(pixel.Col, pixel.Row);
        }

        public void WriteToFiles(string basename)
        {

            for(int i=0;i< firstMoments.Count;i++)
            {
                varianceFactors[i].WriteToFile($"{basename}/factor-tech{i}.exr", 1);
                firstMoments[i].WriteToFile($"{basename}/pixel-tech{i}.exr");
                secondMoments[i].WriteToFile($"{basename}/moment-tech{i}.exr");

                var variance = secondMoments[i] - firstMoments[i].Squared();
                variance.WriteToFile($"{basename}/var-tech{i}.exr");
            }

            var factorRatio = varianceFactors[1] / varianceFactors[0];
            factorRatio.WriteToFile($"{basename}/factor-ratio.exr");
        }

        public bool isReady = false;
        bool UseFilteredFactors = false;
        int numTrainSamples = 0;
        int curIteration = 0;

        // contain candidates
        List<MonochromeImage> candidateFactors;

        List<MonochromeImage> firstMoments;
        List<MonochromeImage> secondMoments;

        MonochromeImage variance;

        protected List<MonochromeImage> varianceFactors;
    }

    protected override void OnPrepareRender()
    {
        base.OnPrepareRender();
        correctionFactors = new CorrectionFactor(NumTrainingSamples, scene.FrameBuffer.Width, scene.FrameBuffer.Height, UseFiltered);
    }

    protected override void OnPostIteration(uint iteration)
    {
        correctionFactors.EndIteration(iteration);
        if (iteration == NumTrainingSamples - 1)
        {
            Logger.Log("Reset Frame!");
            scene.FrameBuffer.Reset();
        }
    }

    protected override void OnPreIteration(uint iteration)
    {
        correctionFactors.StartIteration();
    }

    protected override void OnAfterRender()
    {
        string path = Path.Join(scene.FrameBuffer.Basename);
        correctionFactors.WriteToFiles(path);
    }

    public void OnEstimateVariance(Pixel pixel, RgbColor fVal,
                               int techIdx, Span<float> misweights)
    {
        correctionFactors.Add(techIdx, pixel, fVal);
    }

    protected override RgbColor PerformNextEventEstimation(in SurfaceShader shader, ref PathState state)
    {
        if (scene.Emitters.Count == 0)
            return RgbColor.Black;

        Reservoir<SurfaceSample> reservoir = new Reservoir<SurfaceSample>();
        GenerateNextEvtSamples(shader, ref state, ref reservoir);

        // Get output sample from reservoir
        (SurfaceSample lightSample, RgbColor contrib) = reservoir.GetSample();
        if (reservoir.NotValid())
            return RgbColor.Black;

        if (!scene.Raytracer.IsOccluded(shader.Point, lightSample.Point))
        {
            Vector3 lightToSurface = Vector3.Normalize(shader.Point.Position - lightSample.Point.Position);
            float lightSelectProb = 1.0f / scene.Emitters.Count;

            // Compute the jacobian for surface area -> solid angle
            // (Inverse of the jacobian for solid angle pdf -> surface area pdf)
            float jacobian = SampleWarp.SurfaceAreaToSolidAngle(shader.Point, lightSample.Point);

            // Compute surface area PDFs
            float pdfNextEvt = lightSample.Pdf * lightSelectProb * NumShadowRays;
            float pdfBsdfSolidAngle = DirectionPdf(shader, -lightToSurface, state);
            float pdfBsdf = pdfBsdfSolidAngle * jacobian;

            // Compute the resulting balance heuristic weights
            var c = correctionFactors.Get(state.Pixel);
            float denom = c * pdfBsdf + pdfNextEvt;
            float pdfRatio = pdfNextEvt / denom;
            float misWeight = EnableBsdfDI ? pdfRatio : 1;

            RegisterSample(state.Pixel, contrib * state.PrefixWeight, misWeight,
                state.Depth + 1, true);
            OnNextEventResult(shader, state, misWeight, contrib);
            OnEstimateVariance(state.Pixel, contrib * state.PrefixWeight, 0, [misWeight, 1.0f - misWeight]);
            return misWeight * contrib;
        }

        return RgbColor.Black;
    }

    protected override RgbColor OnLightHit(in Ray ray, in SurfacePoint hit, ref PathState state, Emitter light)
    {
        float misWeight = 1.0f;
        float pdfNextEvt;
        var emission = light.EmittedRadiance(hit, -ray.Direction);
        if (state.Depth > 1)
        { // directly visible emitters are not explicitely connected
          // Compute the solid angle pdf of next event
            var jacobian = SampleWarp.SurfaceAreaToSolidAngle(state.PreviousHit.Value, hit);
            pdfNextEvt = light.PdfUniformArea(hit) / scene.Emitters.Count * NumShadowRays / jacobian;

            // Compute balance heuristic MIS weights
            var c = correctionFactors.Get(state.Pixel);
            float denom = pdfNextEvt + c * state.PreviousPdf;
            misWeight = c * state.PreviousPdf / denom;

            if (!EnableBsdfDI) misWeight = 0;
        }

        RegisterSample(state.Pixel, emission * state.PrefixWeight, misWeight, state.Depth, false);
        OnHitLightResult(ray, state, misWeight, emission, false);
        OnEstimateVariance(state.Pixel, emission * state.PrefixWeight, 1, [1.0f - misWeight, misWeight]);
        return misWeight * emission;
    }


    protected override RgbColor PerformBackgroundNextEvent(in SurfaceShader shader, ref PathState state)
    {
        if (scene.Background == null)
            return RgbColor.Black; // There is no background

        Reservoir<BackgroundSample> reservoir = new Reservoir<BackgroundSample>();
        GenerateBackgroundSamples(shader, ref state, ref reservoir);

        // Get output sample from reservoir
        (BackgroundSample sample, RgbColor contrib) = reservoir.GetSample();
        if (reservoir.NotValid())
            return RgbColor.Black;

        if (scene.Raytracer.LeavesScene(shader.Point, sample.Direction))
        {
            var bsdfTimesCosine = shader.EvaluateWithCosine(sample.Direction);
            var pdfBsdf = DirectionPdf(shader, sample.Direction, state);

            // Prevent NaN / Inf
            if (pdfBsdf == 0 || sample.Pdf == 0)
                return RgbColor.Black;

            float pdfBackground = sample.Pdf * NumShadowRays;

            var c = correctionFactors.Get(state.Pixel);

            // Since the densities are in solid angle unit, no need for any conversions here
            float denom = c * pdfBsdf + (pdfBackground);
            float misWeight = pdfBackground / denom;
            misWeight = EnableBsdfDI ? misWeight : 1;

            Debug.Assert(float.IsFinite(contrib.Average));
            Debug.Assert(float.IsFinite(misWeight));

            RegisterSample(state.Pixel, contrib * state.PrefixWeight, misWeight, state.Depth + 1, true);
            OnNextEventResult(shader, state, misWeight, contrib);
            OnEstimateVariance(state.Pixel, contrib * state.PrefixWeight, 0, [misWeight, 1.0f - misWeight]);
            return misWeight * contrib;
        }
        return RgbColor.Black;
    }

    protected override RgbColor OnBackgroundHit(in Ray ray, ref PathState state)
    {
        if (scene.Background == null || !EnableBsdfDI)
            return RgbColor.Black;

        float misWeight = 1.0f;
        float pdfNextEvent;
        var emission = scene.Background.EmittedRadiance(ray.Direction);
        if (state.Depth > 1)
        {
            var c = correctionFactors.Get(state.Pixel);

            // Compute the balance heuristic MIS weight
            pdfNextEvent = scene.Background.DirectionPdf(ray.Direction) * NumShadowRays;

            float denom = c * state.PreviousPdf + pdfNextEvent;
            misWeight = c * state.PreviousPdf / denom;
        }


        RegisterSample(state.Pixel, emission * state.PrefixWeight, misWeight, state.Depth, false);
        OnHitLightResult(ray, state, misWeight, emission, true);
        OnEstimateVariance(state.Pixel, emission * state.PrefixWeight, 1, [1.0f - misWeight, misWeight]);
        return misWeight * emission;
    }
}