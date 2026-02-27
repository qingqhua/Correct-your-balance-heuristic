
using System.Runtime.CompilerServices;
using VarAwareVCM;
class Program
{
    /// <summary>
    /// Equal time results
    /// </summary>
    static void RunEqualTime()
    {
        List<SceneConfig> scenes = new() {
           //SceneRegistry.LoadScene("RoughGlassesIndirect",maxDepth:10),
           //SceneRegistry.LoadScene("Bookshelf",maxDepth:10),
           //SceneRegistry.LoadScene("VeachBidir",maxDepth:10),
           SceneRegistry.LoadScene("CornellBox",maxDepth:10),
           //SceneRegistry.LoadScene("CornellBoxSpheres",maxDepth:10),
           //SceneRegistry.LoadScene("CornellDuck",maxDepth:10),
           //SceneRegistry.LoadScene("StageNight",maxDepth:10),
           //SceneRegistry.LoadScene("TargetPractice",maxDepth:10),
        };

        int RenderTime = 30;
        int spp = int.MaxValue;

        Benchmark render = new(new VCMExperiment(spp, RenderTime), scenes, 
            $"../../../Results", 640, 480);
        render.Run(skipReference: true);
    }

    static void Main(string[] args)
    {
        string GetThisFilePath([CallerFilePath] string path = null) => path;
        var thisFilePath = Path.GetDirectoryName(GetThisFilePath());
        SceneRegistry.AddSource(Path.Join(thisFilePath, "../Scenes"));
        RunEqualTime();

    }
};

