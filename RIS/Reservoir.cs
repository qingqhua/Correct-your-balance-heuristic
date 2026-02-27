namespace RIS;

public class Reservoir<T>
{
    float wsum = 0f;
    T sample;
    RgbColor target;
    public Reservoir() { }

    public void AddSample(T s, float w, RgbColor target, ref RNG rng)
    {
        wsum += w;
        if (rng.NextFloat() < w / wsum)
        {
            sample = s;
            this.target = target;
        }
    }

    public (T, RgbColor) GetSample()
    {
        var contrib = wsum / target.Average * target;
        return (sample, contrib);
    }

    public float GetPdf()
    {
        var pdf = target.Average / wsum;
        return pdf;
    }

    public float GetContribWeight() { return wsum / target.Average; }
    public RgbColor GetTargetFunction() { return target; }

    public bool NotValid()
    {
        if (target == RgbColor.Black || float.IsNaN(wsum))
            return true;
        return false;
    }

}