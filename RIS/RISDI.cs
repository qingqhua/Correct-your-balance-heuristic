namespace RIS;

/// <summary>
/// Implementation of resampled importance sampling [Talbot et al. 2005; Bitterli et al. 2020]. 
/// On direct illumination with an MIS combination of BSDF and light sampling.
/// </summary>
public class RISDI : PathTracer
{
    /// <summary>
    /// Number of candidates for next event RIS sample.
    /// </summary>
    public int NumNextEvtCandidates = 0;

    public virtual void OnEstimateNormalizationFactor(Pixel pixel, RgbColor target)
    {
    }

    protected override void RenderPixel(uint row, uint col, ref RNG rng)
    {
        uint pixelIndex = (uint)(row * scene.FrameBuffer.Width + col);
        var sampleIndex = (uint)scene.FrameBuffer.CurIteration;
        rng = new(BaseSeed, pixelIndex, sampleIndex);
        base.RenderPixel(row, col, ref rng);
    }

    protected virtual void GenerateNextEvtSamples(in SurfaceShader shader, ref PathState state, ref Reservoir<SurfaceSample> reservoir)
    {
        for (int i = 0; i < NumNextEvtCandidates; i++)
        {
            // Select a light source
            int idx = state.Rng.NextInt(scene.Emitters.Count);
            var light = scene.Emitters[idx];
            float lightSelectProb = 1.0f / scene.Emitters.Count;

            // Sample a point on the light source
            var lightSample = light.SampleUniformArea(state.Rng.NextFloat2D());

            Vector3 lightToSurface = Vector3.Normalize(shader.Point.Position - lightSample.Point.Position);
            float jacobian = SampleWarp.SurfaceAreaToSolidAngle(shader.Point, lightSample.Point);

            // Avoid Inf / NaN
            if (jacobian == 0) continue;

            var pdf = lightSample.Pdf / jacobian * lightSelectProb * NumShadowRays;
            var emission = light.EmittedRadiance(lightSample.Point, lightToSurface);
            var bsdfCos = shader.EvaluateWithCosine(-lightToSurface);

            // weighting of each sample is proportional to target function
            var mis = 1.0f / NumNextEvtCandidates;

            // Our target function is Le * Bsdf * cos, dosent include visibility 
            var target = emission * bsdfCos;

            var w = mis * target.Average / pdf;
            reservoir.AddSample(lightSample,  w, target, ref state.Rng);

            OnEstimateNormalizationFactor(state.Pixel, mis * target / pdf);
        }
    }

    protected virtual void GenerateBackgroundSamples(in SurfaceShader shader, ref PathState state, ref Reservoir<BackgroundSample> reservoir)
    {
        for (int i = 0; i < NumNextEvtCandidates; i++)
        {
            var rng = state.Rng.NextFloat2D();

            var sample = scene.Background.SampleDirection(rng);
            var bsdfTimesCosine = shader.EvaluateWithCosine(sample.Direction);

            // Since the densities are in solid angle unit, no need for any conversions here
            var target = sample.Weight * bsdfTimesCosine / NumShadowRays * sample.Pdf;

            Debug.Assert(float.IsFinite(target.Average));
            var mis = 1.0f / NumNextEvtCandidates;
            var w = mis * target.Average / sample.Pdf;

            reservoir.AddSample(sample, w, target, ref state.Rng);

            OnEstimateNormalizationFactor(state.Pixel, mis * target / sample.Pdf);
        }
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

            // Since the densities are in solid angle unit, no need for any conversions here
            float misWeight = EnableBsdfDI ? 1 / (1.0f + pdfBsdf / (sample.Pdf * NumShadowRays)) : 1;

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
            float pdfRatio = pdfBsdf / pdfNextEvt;
            float misWeight = EnableBsdfDI ? 1.0f / (pdfRatio + 1) : 1;

            RegisterSample(state.Pixel, contrib * state.PrefixWeight, misWeight,
                state.Depth + 1, true);
            OnNextEventResult(shader, state, misWeight, contrib);
            return misWeight * contrib;
        }

        return RgbColor.Black;
    }
}
