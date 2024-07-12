using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Tensorflow.Keras.Engine;


namespace C__yolov8_OnnxRuntime_Demo
{
    public partial class Form2 : Form
    {
        public Form2()
        {
            InitializeComponent();
        }

        string imgFilter = "图片|*.bmp;*.jpg;*.jpeg;*.tiff;*.tiff;*.png";

        YoloV8 yoloV8;
        Mat image;

        string image_path = "";
        string model_path;

        string video_path = "";
        string videoFilter = "视频|*.mp4;*.avi;*.dav";
        VideoCapture vcapture;

        /// <summary>
        /// 单图推理
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button2_Click(object sender, EventArgs e)
        {

            if (image_path == "")
            {
                return;
            }

            button2.Enabled = false;
            pictureBox2.Image = null;
            textBox1.Text = "";

            Application.DoEvents();

            image = new Mat(image_path);

            List<DetectionResult> detResults = yoloV8.Detect(image);

            //绘制结果
            Mat result_image = image.Clone();
            foreach (DetectionResult r in detResults)
            {
                string info = $"{r.Class}:{r.Confidence:P0}";
                //绘制
                Cv2.PutText(result_image, info, new OpenCvSharp.Point(r.Rect.TopLeft.X, r.Rect.TopLeft.Y - 10), HersheyFonts.HersheySimplex, 1, Scalar.Red, 2);
                Cv2.Rectangle(result_image, r.Rect, Scalar.Red, thickness: 2);
            }

            if (pictureBox2.Image != null)
            {
                pictureBox2.Image.Dispose();
            }
            pictureBox2.Image = new Bitmap(result_image.ToMemoryStream());
            textBox1.Text = yoloV8.DetectTime();

            button2.Enabled = true;

        }

        /// <summary>
        /// 窗体加载，初始化
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_Load(object sender, EventArgs e)
        {
            image_path = "test/dog.jpg";
            pictureBox1.Image = new Bitmap(image_path);

            model_path = "model/yolov8n.onnx";

            yoloV8 = new YoloV8(model_path, "model/lable.txt");
        }

        /// <summary>
        /// 选择图片
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button1_Click_1(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = imgFilter;
            if (ofd.ShowDialog() != DialogResult.OK) return;

            pictureBox1.Image = null;

            image_path = ofd.FileName;
            pictureBox1.Image = new Bitmap(image_path);

            textBox1.Text = "";
            pictureBox2.Image = null;
        }

        /// <summary>
        /// 选择视频
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button4_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = videoFilter;
            ofd.InitialDirectory = Application.StartupPath + "\\test";
            if (ofd.ShowDialog() != DialogResult.OK) return;
            video_path = ofd.FileName;
            textBox1.Text = video_path;
        }

        /// <summary>
        /// 视频推理
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button3_Click(object sender, EventArgs e)
        {
            if (video_path == "")
            {
                MessageBox.Show("请先选择视频！");
                return;
            }

            textBox1.Text = "开始检测";
            Application.DoEvents();
            Thread thread = new Thread(new ThreadStart(VideoDetection));
            thread.Start();
            thread.Join();
            textBox1.Text = "检测完成！";
        }

        void VideoDetection()
        {
            vcapture = new VideoCapture(video_path);
            if (!vcapture.IsOpened())
            {
                MessageBox.Show("打开视频文件失败");
                return;
            }

            Mat frame = new Mat();
            List<DetectionResult> detResults;

            // 获取视频的fps
            double videoFps = vcapture.Get(VideoCaptureProperties.Fps);
            // 计算等待时间（毫秒）
            int delay = (int)(1000 / videoFps);
            Stopwatch _stopwatch = new Stopwatch();

            Cv2.NamedWindow("DetectionResult 按下ESC，退出", WindowFlags.Normal);
            Cv2.ResizeWindow("DetectionResult 按下ESC，退出", vcapture.FrameWidth / 2, vcapture.FrameHeight / 2);

            #region 原有不跳帧
            //while (vcapture.Read(frame))
            //{
            //    if (frame.Empty())
            //    {
            //        MessageBox.Show("读取失败");
            //        return;
            //    }
            //    _stopwatch.Restart();

            //    delay = (int)(1000 / videoFps);

            //    detResults = yoloV8.Detect(frame);

            //    //绘制结果
            //    foreach (DetectionResult r in detResults)
            //    {
            //        Cv2.PutText(frame, $"{r.Class}:{r.Confidence:P0}", new OpenCvSharp.Point(r.Rect.TopLeft.X, r.Rect.TopLeft.Y - 10), HersheyFonts.HersheySimplex, 1, Scalar.Red, 2);
            //        Cv2.Rectangle(frame, r.Rect, Scalar.Red, thickness: 2);
            //    }

            //    Cv2.ImShow("DetectionResult 按下ESC，退出", frame);

            //    // for test
            //    // delay = 1;
            //    delay = (int)(delay - _stopwatch.ElapsedMilliseconds);
            //    if (delay <= 0)
            //    {
            //        delay = 1;
            //    }
            //    //Console.WriteLine("delay:" + delay.ToString()) ;
            //    if (Cv2.WaitKey(delay) == 27 || Cv2.GetWindowProperty("DetectionResult 按下ESC，退出", WindowPropertyFlags.Visible) < 1.0)
            //    {
            //        break;
            //    }
            //}

            //Cv2.DestroyAllWindows();
            //vcapture.Release();
            #endregion

            #region 跳帧
            // 设置跳帧间隔，例如每3帧处理一次
            int skipFrames = 6; // 这意味着实际上每3帧处理一次，因为从0开始计数

            bool isReturn = true;
            while (isReturn)
            {
                // 读取帧
                for (int i = 0; i <= skipFrames; i++)
                {
                    if (!vcapture.Read(frame))
                    {
                        isReturn = false;
                        // 视频结束
                        break;
                    }
                    // 跳过不需要处理的帧
                    if (i != skipFrames) continue;
                }

                // 当到达要处理的帧时，执行目标检测或其他处理
                if (frame.Empty())
                    break;

                // 在此处调用你的目标检测函数，比如YOLO检测
                // ProcessFrame(frame);
                delay = (int)(1000 / videoFps);

                detResults = yoloV8.Detect(frame);

                //绘制结果
                foreach (DetectionResult r in detResults)
                {
                    Cv2.PutText(frame, $"{r.Class}:{r.Confidence:P0}", new OpenCvSharp.Point(r.Rect.TopLeft.X, r.Rect.TopLeft.Y - 10), HersheyFonts.HersheySimplex, 1, Scalar.Red, 2);
                    Cv2.Rectangle(frame, r.Rect, Scalar.Red, thickness: 2);
                }

                //// 显示或保存处理后的帧
                //Cv2.ImShow("Processed Frame", frame);

                //// 检查用户按键，用于退出循环
                //if (Cv2.WaitKey(1) >= 0)
                //    break;
                Cv2.ImShow("DetectionResult 按下ESC，退出", frame);

                // for test
                // delay = 1;
                delay = (int)(delay - _stopwatch.ElapsedMilliseconds);
                if (delay <= 0)
                {
                    delay = 1;
                }
                //Console.WriteLine("delay:" + delay.ToString()) ;
                if (Cv2.WaitKey(delay) == 27 || Cv2.GetWindowProperty("DetectionResult 按下ESC，退出", WindowPropertyFlags.Visible) < 1.0)
                {
                    break;
                }
            }

            Cv2.DestroyAllWindows();
            vcapture.Release();
            #endregion
        }


        void TensorFlowVideoDetrction()
        {
            //// 加载YOLO模型
            //var model = new IModel("path_to_your_yolo_model");

            //// 视频读取与处理
            //using (var videoCapture = new VideoCapture("path_to_your_video"))
            //{
            //    var frameCount = (int)videoCapture.Get(CapProp.FrameCount);
            //    for (var i = 0; i < frameCount; i++)
            //    {
            //        using (var frame = videoCapture.Read())
            //        {
            //            // 异步处理帧
            //            var detectionTask = Task.Run(() =>
            //            {
            //                // 预处理图像
            //                var inputImage = PreprocessImage(frame);

            //                // 使用YOLO模型进行检测
            //                var detections = model.Detect(inputImage);

            //                // 后处理并显示结果
            //                DrawDetections(frame, detections);

            //                // 显示或保存处理后的帧
            //                Cv2.ImShow("YOLO Detection", frame);
            //            });

            //            // 等待当前帧处理完成或控制处理速度
            //            detectionTask.Wait(); // 或使用 await detectionTask; 在async方法中
            //        }

            //        // 按需添加延时或帧跳过逻辑
            //        if (Cv2.WaitKey(1) >= 0)
            //            break;
            //    }
            //}
        }

        //保存
        SaveFileDialog sdf = new SaveFileDialog();
        private void button6_Click(object sender, EventArgs e)
        {
            if (pictureBox2.Image == null)
            {
                return;
            }
            Bitmap output = new Bitmap(pictureBox2.Image);
            sdf.Title = "保存";
            sdf.Filter = "Images (*.jpg)|*.jpg|Images (*.png)|*.png|Images (*.bmp)|*.bmp";
            if (sdf.ShowDialog() == DialogResult.OK)
            {
                switch (sdf.FilterIndex)
                {
                    case 1:
                        {
                            output.Save(sdf.FileName, ImageFormat.Jpeg);
                            break;
                        }
                    case 2:
                        {
                            output.Save(sdf.FileName, ImageFormat.Png);
                            break;
                        }
                    case 3:
                        {
                            output.Save(sdf.FileName, ImageFormat.Bmp);
                            break;
                        }
                }
                MessageBox.Show("保存成功，位置：" + sdf.FileName);
            }

        }
    }

}
