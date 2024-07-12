using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace C__yolov8_OnnxRuntime_Demo
{
    public class YoloV8
    {
        float[] input_tensor_data;
        float[] outputData;
        List<DetectionResult> detectionResults;

        int input_height;
        int input_width;

        InferenceSession onnx_session;

        public string[] class_names;
        int class_num;
        int box_num;

        float conf_threshold;
        float nms_threshold;

        float ratio_height;
        float ratio_width;

        public double preprocessTime;
        public double inferTime;
        public double postprocessTime;
        public double totalTime;
        public double detFps;

        public String DetectTime()
        {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.AppendLine($"Preprocess: {preprocessTime:F2}ms");
            stringBuilder.AppendLine($"Infer: {inferTime:F2}ms");
            stringBuilder.AppendLine($"Postprocess: {postprocessTime:F2}ms");
            stringBuilder.AppendLine($"Total: {totalTime:F2}ms");

            return stringBuilder.ToString();
        }

        public YoloV8(string model_path, string classer_path)
        {
            // 创建输出会话，用于输出模型读取信息
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            //options.AppendExecutionProvider_CPU(0);// 设置为CPU上运行

            // 创建推理模型类，读取本地模型文件
            onnx_session = new InferenceSession(model_path, options);

            class_names = File.ReadAllLines(classer_path, Encoding.UTF8);
            class_num = class_names.Length;

            input_height = 640;
            input_width = 640;

            box_num = 8400;

            conf_threshold = 0.5f;
            nms_threshold = 0.5f;

            detectionResults = new List<DetectionResult>();
        }

        void Preprocess(Mat image)
        {
            //图片缩放
            int height = image.Rows;
            int width = image.Cols;
            Mat temp_image = image.Clone();

            if (height > input_height || width > input_width)
            {
                float scale = Math.Min((float)input_height / height, (float)input_width / width);
                OpenCvSharp.Size new_size = new OpenCvSharp.Size((int)(width * scale), (int)(height * scale));
                Cv2.Resize(image, temp_image, new_size);
            }

            ratio_height = (float)height / temp_image.Rows;
            ratio_width = (float)width / temp_image.Cols;
            Mat input_img = new Mat();
            Cv2.CopyMakeBorder(temp_image, input_img, 0, input_height - temp_image.Rows, 0, input_width - temp_image.Cols, BorderTypes.Constant, 0);

            Cv2.CvtColor(input_img, input_img, ColorConversionCodes.BGR2RGB);

           // Cv2.ImShow("Resize", input_img);

            //归一化
            input_img.ConvertTo(input_img, MatType.CV_32FC3, 1.0 / 255);

            input_tensor_data = Common.ExtractMat(input_img);

            input_img.Dispose();
            temp_image.Dispose();
        }

        void Postprocess(float[] outputData)
        {
            detectionResults.Clear();

            float[] data = Common.Transpose(outputData, class_num + 4, box_num);

            float[] confidenceInfo = new float[class_num];
            float[] rectData = new float[4];

            List<DetectionResult> detResults = new List<DetectionResult>();

            for (int i = 0; i < box_num; i++)
            {
                Array.Copy(data, i * (class_num + 4), rectData, 0, 4);
                Array.Copy(data, i * (class_num + 4) + 4, confidenceInfo, 0, class_num);

                float score = confidenceInfo.Max(); // 获取最大值

                int maxIndex = Array.IndexOf(confidenceInfo, score); // 获取最大值的位置

                int _centerX = (int)(rectData[0] * ratio_width);
                int _centerY = (int)(rectData[1] * ratio_height);
                int _width = (int)(rectData[2] * ratio_width);
                int _height = (int)(rectData[3] * ratio_height);

                detResults.Add(new DetectionResult(
                   maxIndex,
                   class_names[maxIndex],
                   new Rect(_centerX - _width / 2, _centerY - _height / 2, _width, _height),
                   score));
            }

            //NMS
            CvDnn.NMSBoxes(detResults.Select(x => x.Rect), detResults.Select(x => x.Confidence), conf_threshold, nms_threshold, out int[] indices);
            detResults = detResults.Where((x, index) => indices.Contains(index)).ToList();

            detectionResults = detResults;
        }

        internal List<DetectionResult> Detect(Mat image)
        {

            var t1 = Cv2.GetTickCount();

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            Preprocess(image);

            preprocessTime = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Restart();

            Tensor<float> input_tensor = new DenseTensor<float>(input_tensor_data, new[] { 1, 3, 640, 640 });
            List<NamedOnnxValue> input_container = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", input_tensor),
            };

            var ort_outputs = onnx_session.Run(input_container).ToArray();

            inferTime = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Restart();

            outputData = ort_outputs[0].AsTensor<float>().ToArray();

            Postprocess(outputData);

            postprocessTime = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Stop();

            totalTime = preprocessTime + inferTime + postprocessTime;

            detFps = (double)stopwatch.Elapsed.TotalSeconds / (double)stopwatch.Elapsed.Ticks;

            var t2 = Cv2.GetTickCount();

            detFps = 1 / ((t2 - t1) / Cv2.GetTickFrequency());

            return detectionResults;

        }

    }
}
