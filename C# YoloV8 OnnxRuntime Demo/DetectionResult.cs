using OpenCvSharp;

namespace C__yolov8_OnnxRuntime_Demo
{

    public class DetectionResult
        {
            public DetectionResult(int ClassId, string Class, Rect Rect, float Confidence)
            {
                this.ClassId = ClassId;
                this.Confidence = Confidence;
                this.Rect = Rect;
                this.Class = Class;
            }

            public string Class { get; set; }

            public int ClassId { get; set; }

            public float Confidence { get; set; }

            public Rect Rect { get; set; }

        }
    

}
