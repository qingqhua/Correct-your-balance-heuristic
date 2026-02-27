namespace RIS;

class EqualTimeExperiment : Experiment
{
    int numIter = int.MaxValue;
    int NumNextEvtCandidates = 32;
    int MaximumRenderTimeMs = 5000;
    int numTrainCorrerctionSamples = 1;
    int blurRadius = 32;

    public EqualTimeExperiment(int time = 5)
    {
        MaximumRenderTimeMs = time * 1000;
    }

    public override List<Method> MakeMethods()
    {
        return new()
        {
            new ($"RIS", new RISDI()
            {
                NumNextEvtCandidates = NumNextEvtCandidates,
                NumShadowRays = 1,
                EnableBsdfDI = true,
                TotalSpp = numIter,
                MaximumRenderTimeMs = MaximumRenderTimeMs,
                EnableDenoiser  = false,
                BaseSeed = 0xC030114,
            }),

            new ($"VarAware", new VarAware()
            {
                NumNextEvtCandidates = NumNextEvtCandidates,
                NumShadowRays = 1,
                NumTrainingSamples = numTrainCorrerctionSamples,
                UseFiltered  = true,
                EnableBsdfDI = true,
                TotalSpp = numIter,
                MaximumRenderTimeMs = MaximumRenderTimeMs,
                EnableDenoiser  = false,
                BaseSeed = 0xC030114,
            }),

            new ($"Ours", new Ours()
            {
                NumNextEvtCandidates = NumNextEvtCandidates,
                NumShadowRays = 1,
                NumTrainingSamples = numTrainCorrerctionSamples,
                UseFiltered  = true,
                BlurRadius = blurRadius,
                EnableBsdfDI = true,
                TotalSpp = numIter,
                MaximumRenderTimeMs = MaximumRenderTimeMs,
                EnableDenoiser  = false,
                BaseSeed = 0xC030114,
            }),

            new ($"Nabata", new Nabata()
            {
                NumNextEvtCandidates = NumNextEvtCandidates,
                NumShadowRays = 1,
                UseFiltered  = false,
                EnableBsdfDI = true,
                TotalSpp = numIter,
                MaximumRenderTimeMs = MaximumRenderTimeMs,
                EnableDenoiser  = false,
                BaseSeed = 0xC030114,
            }),

            new ($"NextEvtRIS", new RISDI()
            {
                NumNextEvtCandidates = NumNextEvtCandidates,
                NumShadowRays = 1,
                EnableBsdfDI = false,
                TotalSpp = numIter,
                MaximumRenderTimeMs = MaximumRenderTimeMs,
                EnableDenoiser  = false,
                BaseSeed = 0xC030114,
            }),
        };
    }

    public override void OnDoneScene(Scene scene, string dir, int minDepth, int maxDepth)
    {
        GenerateRelMSE(scene, dir, minDepth, maxDepth);
        GenerateHTML(scene, dir, minDepth, maxDepth);
    }

    public void GenerateHTML(Scene scene, string dir, int minDepth, int maxDepth)
    {
        // Read json from file
        var refimg = new RgbImage(Path.Join(dir, "Reference.exr"));
        var balance = new RgbImage(Path.Join(dir, "RIS.exr"));
        var varAware = new RgbImage(Path.Join(dir, "VarAware.exr"));
        var ours = new RgbImage(Path.Join(dir, "Ours.exr"));
        var nabata = new RgbImage(Path.Join(dir, "Nabata.exr"));

        var errBal = Metrics.RelMSE(balance, refimg);
        var errVarAware = Metrics.RelMSE(varAware, refimg);
        var errOurs = Metrics.RelMSE(ours, refimg);
        var errNabata = Metrics.RelMSE(nabata, refimg);

        string html = "<!DOCTYPE html><html><head>" + FlipBook.Header;
        html +=
            "<style>" +
            "    table {" +
            "        border-collapse: collapse;" +
            "        width: 100%;" +
            "    }" +
            "    th, td {" +
            "        padding: 8px;" +
            "        text-align: left;" +
            "        border-bottom: 1px solid #DDD;" +
            "    }" +
            "    tr:hover {background-color: #D6EEEE;}" +
            "    body {" +
            "        max-width: 1000px;" +
            "        margin: 0 auto;" +
            "    }" +
            "</style>";

        html += "</head><body>";

        // Tonemapper for better visualization of noise pattern
        var exp = 1.0f;
        if (scene.Name == "RGBSofa")
            exp = -2.0f;
        if (scene.Name == "VeachMIS")
            exp = -3.0f;

        refimg = (RgbImage)SimpleImageIO.Tonemap.Exposure(refimg, exp);
        balance = (RgbImage)SimpleImageIO.Tonemap.Exposure(balance, exp);
        varAware = (RgbImage)SimpleImageIO.Tonemap.Exposure(varAware, exp);
        nabata = (RgbImage)SimpleImageIO.Tonemap.Exposure(nabata, exp);
        ours = (RgbImage)SimpleImageIO.Tonemap.Exposure(ours, exp);

        var layers = new List<KeyValuePair<string, Image>>();
        layers.Add(new KeyValuePair<string, Image>("Reference", (refimg)));
        layers.Add(new KeyValuePair<string, Image>("Balance", (balance)));
        layers.Add(new KeyValuePair<string, Image>("Grittmann et al. 2019", (varAware)));
        layers.Add(new KeyValuePair<string, Image>("Nabata et al. 2020", (nabata)));
        layers.Add(new KeyValuePair<string, Image>("Ours", (ours)));
        ;

        html += "<h3>Equal-time Results (5s)</h3>" + FlipBook.Make(layers, FlipBook.DataType.Float16);

        // Relative MSE Images
        varAware = new RgbImage(Path.Join(dir, "RelMSE", "VarAware.exr"));
        balance = new RgbImage(Path.Join(dir, "RelMSE", "RIS.exr"));
        varAware = new RgbImage(Path.Join(dir, "RelMSE", "VarAware.exr"));
        nabata = new RgbImage(Path.Join(dir, "RelMSE", "Nabata.exr"));
        ours = new RgbImage(Path.Join(dir, "RelMSE", "Ours.exr"));

        layers = new List<KeyValuePair<string, Image>>();
        layers.Add(new KeyValuePair<string, Image>("Balance", (balance)));
        layers.Add(new KeyValuePair<string, Image>("Grittmann et al. 2019", (varAware)));
        layers.Add(new KeyValuePair<string, Image>("Nabata et al. 2020", (nabata)));
        layers.Add(new KeyValuePair<string, Image>("Ours", (ours)));
        ;
        html += "<h3>Relative MSE</h3>" + FlipBook.Make(layers, FlipBook.DataType.Float16);

        // Correction factors images
        var filteredFactors = new RgbImage(Path.Join(dir, "Ours", "correction.exr"));

        layers = new List<KeyValuePair<string, Image>>();
        layers.Add(new KeyValuePair<string, Image>("Ours filtered", (filteredFactors)));

        html += "<h3>Correction factors</h3>" + FlipBook.Make(layers, FlipBook.DataType.Float16);


        File.WriteAllText(dir + ".html", html);
    }

    public void GenerateRelMSE(Scene scene, string dir, int minDepth, int maxDepth)
    {
        var refImg = new RgbImage(Path.Join(dir, "Reference.exr"));
        var ris = new RgbImage(Path.Join(dir, "RIS.exr"));
        var ours = new RgbImage(Path.Join(dir, "Ours.exr"));
        var nabata = new RgbImage(Path.Join(dir, "Nabata.exr"));
        var varAware = new RgbImage(Path.Join(dir, "VarAware.exr"));

        (_, var legend) = HistogramRenderer.Render(ours, ours.Width, ours.Height);
        var tonemapper = new FalseColor(new LinearColormap(0, 0.2f));
        if(scene.Name=="Garage")
            tonemapper = new FalseColor(new LinearColormap(0, 1.5f));
        else if (scene.Name == "ModernHall")
            tonemapper = new FalseColor(new LinearColormap(0, 0.6f));
        var relMse = Metrics.RelMSEImage(ris, refImg);
        var buffer = Metrics.RelMSEImage(ris, refImg);
        
        SimpleImageIO.Filter.Gauss(relMse, buffer, 1);
        relMse = tonemapper.Apply(buffer);
        relMse.WriteToFile(Path.Join(dir, "RelMSE", "RIS.exr"));

        relMse = Metrics.RelMSEImage(ours, refImg);
        SimpleImageIO.Filter.Gauss(relMse, buffer, 1);
        relMse = tonemapper.Apply(buffer);
        relMse.WriteToFile(Path.Join(dir, "RelMSE", "Ours.exr"));

        relMse = Metrics.RelMSEImage(varAware, refImg);
        SimpleImageIO.Filter.Gauss(relMse, buffer, 1);
        relMse = tonemapper.Apply(buffer);
        relMse.WriteToFile(Path.Join(dir, "RelMSE", "VarAware.exr"));

        relMse = Metrics.RelMSEImage(nabata, refImg);
        SimpleImageIO.Filter.Gauss(relMse, buffer, 1);
        relMse = tonemapper.Apply(buffer);
        relMse.WriteToFile(Path.Join(dir, "RelMSE", "Nabata.exr"));
    }
}