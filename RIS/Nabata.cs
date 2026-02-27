namespace RIS;

/// <summary>
/// Implementation of the paper:
/// "Resampling-aware Weighting Functions for Bidirectional Path Tracing Using Multiple Light Sub-Paths." [Nabata et al. 2020]
/// </summary>
public class Nabata : RISDI
{
    public NormalizationFactor Q;
    public int NumTrainNormalizationSamples = 1;
    public bool UseFiltered = true;
    public class NormalizationFactor
    {
        public NormalizationFactor(int numTrainSamples, int width, int height, bool useFilteredFactors = true)
        {
            normalizationFactorCurIt = new(width, height); 
            normalizationFactorPrev = new(width, height); 
            this.numTrainSamples = numTrainSamples;
        }

        public void StartIteration()
        {
            curIteration++;

            // Scale values of the previous iteration to account for having more samples
            if (curIteration > 1)
            {
                normalizationFactorCurIt.Scale((curIteration - 1.0f) / curIteration);
            }
        }

        public void EndIteration(uint iteration)
        {

            Parallel.For(0, normalizationFactorCurIt.Height, row =>
            {
                for (int col = 0; col < normalizationFactorCurIt.Width; ++col)
                    normalizationFactorPrev[col, row] = normalizationFactorCurIt[col, row];
            });

            isReady = true;
        }

        public void Add(Pixel pixel, RgbColor target)
        {

            float v = target.Average;

            normalizationFactorCurIt.AtomicAdd(pixel.Col, pixel.Row, v / curIteration);
        }

        public float Get(Pixel pixel)
        {
            if (!isReady) return 0.0f;
            return normalizationFactorPrev.GetPixel(pixel.Col, pixel.Row);
        }

        public bool IsReady(Pixel pixel)
        {
            return isReady;
        }

        // factor is valid if it's >0 
        public bool IsValid(Pixel pixel)
        {
            return normalizationFactorPrev.GetPixel(pixel.Col, pixel.Row) > 0;
        }

        public void WriteToFiles(string basename)
        {
            if(isReady)
                normalizationFactorPrev.WriteToFile($"{basename}/normalization.exr", 1);
        }

        bool isReady = false;
        int numTrainSamples = 0;
        int curIteration = 0;

        // contain candidates
        MonochromeImage normalizationFactorCurIt;
        MonochromeImage normalizationFactorPrev;

    }

    protected override void OnPrepareRender()
    {
        base.OnPrepareRender();

        Logger.Log("Use" + NumTrainNormalizationSamples + " samples for normalization factor");
        Q = new NormalizationFactor(NumTrainNormalizationSamples, scene.FrameBuffer.Width, scene.FrameBuffer.Height, UseFiltered);
    }

    protected override void OnPostIteration(uint iteration)
    {
        Q.EndIteration(iteration);
        if (iteration == NumTrainNormalizationSamples - 1)
        {
            Logger.Log("Reset Frame!");
            scene.FrameBuffer.Reset();
        }
    }

    protected override void OnPreIteration(uint iteration)
    {
        Q.StartIteration();
    }

    protected override void OnAfterRender()
    {
        string path = Path.Join(scene.FrameBuffer.Basename);
        Q.WriteToFiles(path);
    }

    public override void OnEstimateNormalizationFactor(Pixel pixel, RgbColor target)
    {
        if (float.IsNaN(target.Average) || float.IsInfinity(target.Average)) return;
        Q.Add(pixel, target);
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

            float pdfBackground = sample.Pdf;
            // Re-evaluate ris pdf using Nabata et al. 2020 Eqn 14.
            // If factor is not valid or not ready, we use only candidate pdf for mis
            if (Q.IsReady(state.Pixel) && Q.IsValid(state.Pixel))
            {
                // Nabata et al: inteolate between target and candidate pdf
                var normalizedTarget = reservoir.GetTargetFunction().Average / Q.Get(state.Pixel);
                float pdfNextEvtRis = (1.0f / NumNextEvtCandidates) / sample.Pdf;
                pdfNextEvtRis += (1.0f - 1 / NumNextEvtCandidates) / normalizedTarget;

                pdfBackground = 1.0f / pdfNextEvtRis;

                Debug.Assert(float.IsFinite(pdfBackground));
            }

            // Since the densities are in solid angle unit, no need for any conversions here
            float misWeight = EnableBsdfDI ? 1 / (1.0f + pdfBsdf / (pdfBackground * NumShadowRays)) : 1;

            Debug.Assert(float.IsFinite(contrib.Average));
            Debug.Assert(float.IsFinite(misWeight));

            RegisterSample(state.Pixel, contrib * state.PrefixWeight, misWeight, state.Depth + 1, true);
            OnNextEventResult(shader, state, misWeight, contrib);
            return misWeight * contrib;
        }
        return RgbColor.Black;
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

        // Visibility test
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

            // Re-evaluate ris pdf using Nabata et al. 2020 Eqn 14.
            // If factor is not valid or not ready, we use only candidate pdf for mis
            if (Q.IsReady(state.Pixel) && Q.IsValid(state.Pixel))
            {
                // Nabata et al: inteolate between target and candidate pdf
                var normalizedTarget = reservoir.GetTargetFunction().Average / Q.Get(state.Pixel);
                float pdfNextEvtRis = (1.0f / NumNextEvtCandidates) / pdfNextEvt;
                pdfNextEvtRis += (1.0f - 1 / NumNextEvtCandidates) / normalizedTarget;

                pdfNextEvt = 1.0f / pdfNextEvtRis;

                Debug.Assert(float.IsFinite(pdfNextEvt));
            }

            // Compute the resulting balance heuristic weights
            float denom = pdfBsdf + pdfNextEvt;
            float pdfRatio = pdfNextEvt / denom;
            float misWeight = EnableBsdfDI ? pdfRatio : 1;

            Debug.Assert(!float.IsNaN(misWeight));

            RegisterSample(state.Pixel, contrib * state.PrefixWeight, misWeight,
                state.Depth + 1, true);
            OnNextEventResult(shader, state, misWeight, contrib);
            return misWeight * contrib;
        }
        return RgbColor.Black;
    }

    protected override RgbColor OnLightHit(in Ray ray, in SurfacePoint hit, ref PathState state, Emitter light)
    {
        float misWeight = 1.0f;
        float pdfNextEvt = 1.0f;
        var emission = light.EmittedRadiance(hit, -ray.Direction);
        var target = RgbColor.Black;
        if (state.Depth > 1)
        { // directly visible emitters are not explicitely connected
          // Compute the solid angle pdf of next event
            var jacobian = SampleWarp.SurfaceAreaToSolidAngle(state.PreviousHit.Value, hit);
            pdfNextEvt = light.PdfUniformArea(hit) / scene.Emitters.Count * NumShadowRays / jacobian;

            // Evaluate target function: bsdfcos * emission for this sample
            target = emission * state.PrefixWeight * state.PreviousPdf;

            // Nabata et al. 2020 Eqn 14.
            if (Q.IsReady(state.Pixel) && Q.IsValid(state.Pixel))
            {
                // Re-evaluate pdf for next evt ris
                var normalizedTarget = target.Average / jacobian / Q.Get(state.Pixel);
                float pdfNextEvtRis = (1.0f / NumNextEvtCandidates) / pdfNextEvt;
                pdfNextEvtRis += (1.0f - 1.0f / NumNextEvtCandidates) / normalizedTarget;

                pdfNextEvt = 1.0f / pdfNextEvtRis;
            }

            // Compute balance heuristic MIS weights
            float denom = pdfNextEvt + state.PreviousPdf;
            misWeight = state.PreviousPdf / denom;

            Debug.Assert(!float.IsNaN(misWeight));
            if (!EnableBsdfDI) misWeight = 0;
        }

        RegisterSample(state.Pixel, emission * state.PrefixWeight, misWeight, state.Depth, false);
        OnHitLightResult(ray, state, misWeight, emission, false);
        return misWeight * emission;
    }

    protected override RgbColor OnBackgroundHit(in Ray ray, ref PathState state)
    {
        if (scene.Background == null || !EnableBsdfDI)
            return RgbColor.Black;

        var emission = scene.Background.EmittedRadiance(ray.Direction);
        var target = RgbColor.Black;
        float misWeight = 1.0f;
        float pdfNextEvent;
        if (state.Depth > 1)
        {
            target = emission * state.PrefixWeight * state.PreviousPdf;
            pdfNextEvent = scene.Background.DirectionPdf(ray.Direction) * NumShadowRays;

            if (Q.IsReady(state.Pixel) && Q.IsValid(state.Pixel))
            {
                // Re-evaluate pdf for next evt ris
                var normalizedTarget = target.Average / Q.Get(state.Pixel);
                float pdfNextEvtRis = (1.0f / NumNextEvtCandidates) / pdfNextEvent;
                pdfNextEvtRis += (1.0f - 1.0f / NumNextEvtCandidates) / normalizedTarget;

                pdfNextEvent = 1.0f / pdfNextEvtRis;
            }

            // Compute the balance heuristic MIS weight
            misWeight = 1 / (1 + pdfNextEvent / state.PreviousPdf);
        }

        RegisterSample(state.Pixel, emission * state.PrefixWeight, misWeight, state.Depth, false);
        OnHitLightResult(ray, state, misWeight, emission, true);
        return misWeight * emission;
    }
}