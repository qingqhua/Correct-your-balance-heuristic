namespace SeeSharp.Integrators;

/// <summary>
/// A classic path tracer with next event estimation. Additional per-path user data can be tracked via the
/// generic type provided.
/// </summary>

public class PathStateUserData {
    public RgbColor color { get; set; }
    public int @int { get; set; }
}

public class PathTracerUserData : PathTracerBase<PathStateUserData> {
    protected override RgbColor EstimateIncidentRadiance(Ray ray, ref PathState state) {
        RgbColor radianceEstimate = RgbColor.Black;

        while (state.Depth <= MaxDepth) {
            var hit = scene.Raytracer.Trace(ray);

            // Did the ray leave the scene?
            if (!hit) {
                if (state.Depth >= MinDepth)
                    radianceEstimate += state.PrefixWeight * OnBackgroundHit(ray, ref state);
                break;
            }

            OnHit(ray, hit, ref state);

            SurfaceShader shader = new(hit, -ray.Direction, false);

            if (state.Depth == 1 && EnableDenoiser) {
                var albedo = shader.GetScatterStrength();
                denoiseBuffers.LogPrimaryHit(state.Pixel, albedo, hit.ShadingNormal);
            }

            // Check if a light source was hit.
            Emitter light = scene.QueryEmitter(hit);
            if (light != null && state.Depth >= MinDepth) {
                radianceEstimate += state.PrefixWeight * OnLightHit(ray, hit, ref state, light);
            }

            // Path termination with Russian roulette
            float survivalProb = ComputeSurvivalProbability(ray, hit, state);
            if (state.Rng.NextFloat() > survivalProb || state.Depth == MaxDepth)
                break;

            // Perform next event estimation
            if (state.Depth + 1 >= MinDepth) {
                RgbColor nextEventContrib = RgbColor.Black;
                for (int i = 0; i < NumShadowRays; ++i) {
                    nextEventContrib += PerformBackgroundNextEvent(shader, ref state);
                    nextEventContrib += PerformNextEventEstimation(shader, ref state);
                }
                radianceEstimate += state.PrefixWeight * nextEventContrib / survivalProb;
            }

            // Sample a direction to continue the random walk
            (ray, float bsdfPdf, var bsdfSampleWeight, var approxReflectance) = SampleDirection(shader, state);
            if (bsdfPdf == 0 || bsdfSampleWeight == RgbColor.Black)
                break;

            // Recursively estimate the incident radiance and log the result
            state.PrefixWeight *= bsdfSampleWeight / survivalProb;
            state.ApproxThroughput *= approxReflectance / survivalProb;
            state.Depth++;
            state.PreviousHit = hit;
            state.PreviousPdf = bsdfPdf * survivalProb;
            
            state.UserData.@int = 1;
            state.UserData.color = new RgbColor(1,0,0);

            if (state.Depth == 2) {
                state.UserData.color = new RgbColor(0, 1, 0);
                Console.WriteLine(state.UserData.color);
            }
                

        }

        return radianceEstimate;
    }

    protected override void RenderPixel(uint row, uint col, ref RNG rng) {
        // Sample a ray from the camera
        var offset = rng.NextFloat2D();
        var pixel = new Vector2(col, row) + offset;
        Ray primaryRay = scene.Camera.GenerateRay(pixel, ref rng).Ray;

        PathState state = new() {
            Pixel = new((int)col, (int)row),
            Rng = ref rng,
            PrefixWeight = RgbColor.White,
            ApproxThroughput = RgbColor.White,
            Depth = 1,
            UserData = new PathStateUserData(),
        };

        OnStartPath(ref state);
        var estimate = EstimateIncidentRadiance(primaryRay, ref state);
        OnFinishedPath(estimate, ref state);

        scene.FrameBuffer.Splat(state.Pixel, estimate);
    }

}