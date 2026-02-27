namespace VarAwareVCM;

/// <summary>
/// Main VCM application. Apply the correction factor on merging, also apply separate correction factors for DI (both variance-aware and ours)
/// </summary> 
class VCMExperiment : Experiment
{
    int numIter = int.MaxValue;

    bool EnableLightTracer = true;
    bool EnableHit = true;
    int RenderTime = 30 * 1000;
    int NumShadowRays = 1;
    int NumConnections = 1;

    public VCMExperiment(int numIter, int RenderTime) {
        this.numIter = numIter;
        this.RenderTime = RenderTime * 1000;
    }

    public override List<Method> MakeMethods()
    {
        List<float> merge = new List<float>();
        merge.Add(1.0f);
        merge.Add(0.01f);
        merge.Add(0.1f);
        merge.Add(0.5f);

        List<float> lt = new List<float>();
        {
            lt.Add(1.0f);
            lt.Add(0.01f);
            lt.Add(0.1f);
            lt.Add(0.5f);
        }

        return new()
        {
        new ($"Balance", new VertexConnectionAndMerging()
        {
            NumIterations = numIter,
            NumShadowRays = NumShadowRays,
            NumConnections = NumConnections,
            EnableMerging = true,
            EnableHitting = EnableHit,
            EnableLightTracer = EnableLightTracer,
            DisableCorrelAwareMIS = true,
            MaximumRenderTimeMs = RenderTime,
            RenderTechniquePyramid = false,
        }),       

        new ($"CorrelAware", new VertexConnectionAndMerging()
        {
            NumIterations = numIter,
            NumShadowRays = NumShadowRays,
            NumConnections = NumConnections,
            EnableMerging = true,
            EnableHitting = EnableHit,
            EnableLightTracer = EnableLightTracer,
            DisableCorrelAwareMIS = false,
            MaximumRenderTimeMs = RenderTime,
        }),

         new ($"Ours", new Ours()
        {
            NumIterations = numIter,
            NumShadowRays = NumShadowRays,
            NumConnections = NumConnections,
            EnableMerging = true,
            EnableHitting = EnableHit,
            EnableLightTracer = EnableLightTracer,
            EnableLightTracerFactor = true,
            NumTrainSamples =  1,
            DisableCorrelAwareMIS = false,
            CandidatesMerge = merge.ToArray(),
            CandidatesLightTracer = lt.ToArray(),
            MaximumRenderTimeMs = RenderTime,

        }),

        new ($"VarAware", new VarAwareVCM()
        {
            NumIterations = numIter,
            NumShadowRays = NumShadowRays,
            NumConnections = NumConnections,
            EnableMerging = true,
            EnableHitting = EnableHit,
            NumTrainSamples =  1,
            UseFilteredFactors= true,
            EnableLightTracer = EnableLightTracer,
            EnableLightTracerFactor = true,
            EnableNextEvtFactor = true,
            EnableHitFactor = true,
            DisableCorrelAwareMIS = true,
            MaximumRenderTimeMs = RenderTime,

        }),
        };
    }

    public override void OnDoneScene(Scene scene, string dir, int minDepth, int maxDepth)
    {
        GenerateHtml(scene, dir, minDepth, maxDepth);
    }

    // Note by defualt this function uses outlier rejection, what we use in the paper is relmse
    void GenerateHtml(Scene scene, string dir, int minDepth, int maxDepth)
    {
        var stopwatch = Stopwatch.StartNew();

        string refPath = $"{dir}/Reference.exr";
        RgbImage reference = File.Exists(refPath) ? new(refPath) : null;

        // Create a flip viewer with all the rendered images and the reference (if available)
        var flip = FlipBook.New.SetZoom(FlipBook.InitialZoom.Fit).SetToneMapper(FlipBook.InitialTMO.Exposure(scene.RecommendedExposure));
        float maxError = 0.0f;
        List<float> errors = [];
        List<(string, Image)> errorImages = [];
        List<(string, Image)> squaredErrorImages = [];
        List<string> methods = [ "Balance", "VarAware", "CorrelAware", "Ours" ];
        foreach (string method in methods)
        {
            RgbImage img = new($"{dir}/{method}.exr");
            flip.Add(method, img, FlipBook.DataType.Float16);
        }
        if (reference != null)
        {
            flip.Add("Reference", reference);

            // Ensure valid .json if the error is NaN or Inf
            if (!float.IsFinite(maxError))
                maxError = 1.0f;
        }

        // Assemble html code
        flip.SetToolVisibility(false);
        string htmlBody = "<h3>Equal-time Results (30s)</h3>";
        htmlBody += $"""<div style="display: flex;">{flip.Resize(900, 800)}""";
        htmlBody += "</div>";

        // Relative MSE Images
        flip = FlipBook.New.SetZoom(FlipBook.InitialZoom.Fit).SetToneMapper(FlipBook.InitialTMO.Exposure(scene.RecommendedExposure));
        flip.AddAll(squaredErrorImages);

        //htmlBody += "<h3>False color maps of relative MSE</h3>";
        //htmlBody += $"""<div style="display: flex;">{flip.Resize(900, 800)}""";
        //htmlBody += "</div>";

        // Show speedup numbders
        //htmlBody += "<h3>Statistics</h3>";
        //htmlBody += HtmlUtil.MakeTable(tableRows, true);

        string tableStyle = """
        <style>
            table {
                border-collapse: collapse;
            }
            td, th {
                border: none;
                padding: 4px;
            }
            tr:hover { background-color: #e7f2f1; }
            th {
                padding-top: 6px;
                padding-bottom: 6px;
                text-align: left;
                background-color: #4a96af;
                color: white;
                font-size: smaller;
            }
        </style>
        """;

        var html = HtmlUtil.MakeHTML(FlipBook.Header + tableStyle, htmlBody);
        var layers = new List<KeyValuePair<string, Image>>();
        layers.Add(new KeyValuePair<string, Image>("Filtered Merging (GI)",  new MonochromeImage(Path.Join(dir, "Ours", "variance-factors-merge-filtered.exr"))));
        layers.Add(new KeyValuePair<string, Image>("Filtered Light tracer (DI)",  new MonochromeImage(Path.Join(dir, "Ours", "variance-factors-light-tracer-filtered.exr"))));

        html += "<h3>Our correction factors</h3>" + FlipBook.Make(layers, FlipBook.DataType.Float16);
        File.WriteAllText(dir + ".html", html);

        Logger.Log($"Assembling {scene.Name}.html took {stopwatch.ElapsedMilliseconds}ms");
    }
}