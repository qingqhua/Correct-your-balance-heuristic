using System.Runtime.CompilerServices;
using RIS;

string GetThisFilePath([CallerFilePath] string path = null) => path;
var thisFilePath = Path.GetDirectoryName(GetThisFilePath());
SceneRegistry.AddSource(Path.Join(thisFilePath, "../Scenes"));

// Main results under equal-time comparison; outputs an HTML report and the rendered images.
{
    List<SceneConfig> scenes = new()
{
    SceneRegistry.LoadScene("Garage", maxDepth: 2),
    SceneRegistry.LoadScene("ModernHall", maxDepth: 2),
    SceneRegistry.LoadScene("VeachMIS", maxDepth: 2),
    SceneRegistry.LoadScene("RGBSofa", maxDepth: 2),
};

    Benchmark benchmark = new(new EqualTimeExperiment(), scenes
        , $"../../../Results/", 640, 480);
    benchmark.Run(skipReference: true);
}

