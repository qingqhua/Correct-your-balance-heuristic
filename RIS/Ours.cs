namespace RIS;

/// <summary>
/// Implementation of the paper:
/// "Correct your balance heuristic: Optimizing balance-style multiple importance sampling weights" [Hua et al. 2025]
/// </summary>
public class Ours : RISDI
{
    /// <summary>
    /// Filter the correction factor
    /// </summary>
    public bool UseFiltered = true;

    /// <summary>
    /// Determine how many samples we want to use to compute factor.
    /// </summary>
    public int NumTrainingSamples = 1;

    /// <summary>
    /// Determine how large the radius for filtering second moment.
    /// </summary>
    public int BlurRadius = 32;

    public CorrectionFactor correctionFactors;

    public class CorrectionFactor
    {
        public CorrectionFactor(ReadOnlySpan<float> c, int numTrainSamples, int width, int height, bool useFilteredFactors, int blurRadius)
        {
            int numCandidates = c.Length;
            candidatesSecondMomentPerIt = new(numCandidates); // # candidates x techs
            candidatesSecondMoment = new(numCandidates); // # candidates x techs
            candidatesFirstMoment = new(numCandidates); // # candidates x techs
            candidateFactors = new(numCandidates); // # candidates x techs

            correctionFactors = new(width, height); // # we only have two techniques, thus one ratio
            variance = new(width, height); // # 1
            variance.Fill(float.MaxValue);
            correctionFactors.Fill(1f);

            for (int i = 0; i < numCandidates; i++)
            {
                candidatesSecondMomentPerIt.Add(new(width, height));
                candidatesSecondMoment.Add(new(width, height));
                candidatesFirstMoment.Add(new(width, height));
                candidateFactors.Add(new(width, height));

                candidateFactors[^1].Fill(c[i]);
            }

            // switch between accurate and filtered factors. The later only need 1 iter to estimate.
            this.UseFilteredFactors = useFilteredFactors;
            this.numTrainSamples = numTrainSamples;
            this.blurRadius = blurRadius;
        }

        public void StartIteration()
        {
            curIteration++;

            if (isReady) return;

            // Scale values of the previous iteration to account for having more samples
            if (curIteration > 1)
            {
                Parallel.For(0, candidateFactors.Count, k => // candidates
                {
                    candidatesSecondMoment[k].Scale((curIteration - 1.0f) / curIteration);
                    candidatesSecondMomentPerIt[k].Scale(0);
                    candidatesFirstMoment[k].Scale((curIteration - 1.0f) / curIteration);
                });
            }
        }

        void ComputeFilteredFactors()
        {
            int width = candidatesSecondMomentPerIt[0].Width;
            int height = candidatesSecondMomentPerIt[0].Height;
            MonochromeImage pixelBuffer = new(width, height);
            MonochromeImage varianceBuffer = new(width, height);
            // Compute the variance factors for use in the next iteration
            for (int i = 0; i < candidatesSecondMomentPerIt.Count; ++i) // candidates
            {
                Debug.Assert(curIteration == 1);


                Filter.RepeatedBox(candidatesSecondMoment[i], varianceBuffer, blurRadius);
                Filter.RepeatedBox(candidatesFirstMoment[i], pixelBuffer, 16);

                Parallel.For(0, height, row =>
                {
                    for (int col = 0; col < width; ++col)
                    {
                        var pixelSqr = pixelBuffer[col, row] * pixelBuffer[col, row];
                        var variance = varianceBuffer[col, row] / pixelSqr;
                        // Compare with the blurred variance and get the minimum
                        if (variance < this.variance[col, row] && variance > 0)
                        {
                            this.variance[col, row] = variance;
                            var c = candidateFactors[i].GetPixel(col, row);
                            correctionFactors.SetPixel(col, row, c);
                            Debug.Assert(float.IsFinite(c));
                        }
                    }
                });
            }

            // Apply a wide filter to the ratio image, too
            Filter.RepeatedBox(correctionFactors, correctionFactors, 16);
        }

        void ComputeAccurateFactors()
        {
            int width = candidatesSecondMomentPerIt[0].Width;
            int height = candidatesSecondMomentPerIt[0].Height;
            // Compute the variance factors for use in the next iteration
            for (int k = 0; k < candidatesSecondMomentPerIt.Count; ++k) // candidates
            {
                Parallel.For(0, height, row =>
                {
                    for (int col = 0; col < width; ++col)
                    {
                        var moment = candidatesSecondMoment[k][col, row];
                        var firstMomentSqr = candidatesFirstMoment[k][col, row] * candidatesFirstMoment[k][col, row];
                        var curvariance = moment; // variance for each candidates

                        if ((curvariance < variance[col, row]) && curvariance > 0)
                        {
                            this.variance[col, row] = curvariance;
                            var c = candidateFactors[k].GetPixel(col, row);
                            correctionFactors.SetPixel(col, row, c);
                            Debug.Assert(float.IsFinite(c));
                        }
                    }
                });
            }
        }

        public void EndIteration(uint iteration)
        {
            if (isReady) return;

            Parallel.For(0, candidatesSecondMomentPerIt[0].Height, row =>
            {
                for (int col = 0; col < candidatesSecondMomentPerIt[0].Width; ++col)
                {
                    for (int k = 0; k < candidatesSecondMomentPerIt.Count; ++k)
                        candidatesSecondMoment[k][col, row] += candidatesSecondMomentPerIt[k][col, row] * candidatesSecondMomentPerIt[k][col, row] / curIteration;
                }
            });

            if (iteration == numTrainSamples -1)
            {
                Logger.Log("Solved!");
                if (UseFilteredFactors)
                    ComputeFilteredFactors();
                else
                    ComputeAccurateFactors();
                isReady = true;
            }
        }

        public void Add(int techIdx, Pixel pixel, RgbColor fVal, Span<float> misweights)
        {
            if (isReady) return;

            // iterate over all candiates to get estimation of second moments and first moments for each.
            for (int k = 0; k < candidateFactors.Count; k++)
            {
                var c = new float[2] { 1.0f, candidateFactors[k].GetPixel(pixel.Col, pixel.Row) };

                var nom = c[techIdx] * misweights[techIdx];
                var denom = 0.0f;

                for (int i = 0; i < misweights.Length; i++)
                    denom += c[i] * misweights[i];
                var w = nom / denom;

                Debug.Assert(float.IsFinite(w));
                var v = w * fVal.Average;
                candidatesSecondMomentPerIt[k].AtomicAdd(pixel.Col, pixel.Row, v);
                candidatesFirstMoment[k].AtomicAdd(pixel.Col, pixel.Row, v / curIteration);
            }
        }

        public float Get(Pixel pixel)
        {
            if (!isReady) return 1.0f;
            return correctionFactors.GetPixel(pixel.Col, pixel.Row);
        }

        public void WriteToFiles(string basename)
        {
            correctionFactors.WriteToFile($"{basename}/correction.exr", 1);

            for (int i = 0; i < candidateFactors.Count; i++)
            {
                var c = candidateFactors[i][0, 0];
                candidatesSecondMoment[i].WriteToFile(Path.Join(basename, $"sec-moments-{c}.exr"));
                candidatesFirstMoment[i].WriteToFile(Path.Join(basename, $"pixel-{c}.exr"));
            }

            // Filtered
            for (int i = 0; i < candidateFactors.Count; i++)
            {
                Filter.RepeatedBox(candidatesSecondMoment[i], candidatesSecondMoment[i], blurRadius);
                Filter.RepeatedBox(candidatesFirstMoment[i], candidatesFirstMoment[i], blurRadius);

                var c = candidateFactors[i][0, 0];
                candidatesSecondMoment[i].WriteToFile(Path.Join(basename, $"filtered-sec-moments-{c}.exr"));
            }
        }

        public bool isReady = false;
        bool UseFilteredFactors = false;
        bool UseFullVar = false;
        int numTrainSamples = 0;
        int curIteration = 0;
        int blurRadius = 0;

        // contain candidates
        List<MonochromeImage> candidatesSecondMomentPerIt;
        List<MonochromeImage> candidatesSecondMoment;
        List<MonochromeImage> candidatesFirstMoment;
        List<MonochromeImage> candidateFactors;

        MonochromeImage variance;
        MonochromeImage correctionFactors;
    }

    protected override void OnPrepareRender()
    {
        base.OnPrepareRender();

        Logger.Log("Use" + NumTrainingSamples + " samples for correction factor");

        Span<float> c = [0.00001f, 0.0001f, 0.001f, 0.01f, 0.02f, 0.04f, 0.06f, 0.1f, 0.3f, 0.7f, 1.0f, 10.0f, 100.0f, int.MaxValue];

        if (UseFiltered)
        {
            c = [0.01f, 0.1f, 0.5f, 1.0f];
            if (scene.Background is EnvironmentMap && scene.Emitters.Count == 0)
                c = [0.01f, 0.1f, 0.5f, 1.0f, 2.0f, 10.0f, 100.0f];
        }

        correctionFactors = new CorrectionFactor(c, NumTrainingSamples, scene.FrameBuffer.Width, scene.FrameBuffer.Height, UseFiltered, BlurRadius);
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
        correctionFactors.Add(techIdx, pixel, fVal, misweights);
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


            // Avoid Inf / NaN
            if (jacobian == 0) return RgbColor.Black;

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
        float pdfNextEvt = 1.0f;
        var emission = light.EmittedRadiance(hit, -ray.Direction);
        if (state.Depth > 1)
        { // directly visible emitters are not explicitely connected
          // Compute the solid angle pdf of next event
            var jacobian = SampleWarp.SurfaceAreaToSolidAngle(state.PreviousHit.Value, hit);
            if(jacobian==0) return RgbColor.Black;
            pdfNextEvt = light.PdfUniformArea(hit) / scene.Emitters.Count * NumShadowRays / jacobian;

            // Compute balance heuristic MIS weights
            var c = correctionFactors.Get(state.Pixel);
            float denom = pdfNextEvt + c * state.PreviousPdf;
            misWeight = c * state.PreviousPdf / denom;
            if (!EnableBsdfDI) misWeight = 0;
            OnEstimateVariance(state.Pixel, emission * state.PrefixWeight, 1, [1.0f-misWeight, misWeight]);
        }

        
        RegisterSample(state.Pixel, emission * state.PrefixWeight, misWeight, state.Depth, false);
        OnHitLightResult(ray, state, misWeight, emission, false);
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