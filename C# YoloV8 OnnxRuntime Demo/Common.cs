using OpenCvSharp;
using System;
using System.Runtime.InteropServices;

namespace C__yolov8_OnnxRuntime_Demo
{
    internal class Common
    {

        public static float[] ExtractMat(Mat src)
        {
            OpenCvSharp.Size size = src.Size();
            int channels = src.Channels();
            float[] result = new float[size.Width * size.Height * channels];
            GCHandle resultHandle = default;
            try
            {
                resultHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
                IntPtr resultPtr = resultHandle.AddrOfPinnedObject();
                for (int i = 0; i < channels; ++i)
                {
                    Mat cmat = new Mat(
                       src.Height, src.Width,
                       MatType.CV_32FC1,
                       resultPtr + i * size.Width * size.Height * sizeof(float));

                    Cv2.ExtractChannel(src, cmat, i);

                    cmat.Dispose();

                }
            }
            finally
            {
                resultHandle.Free();
            }

            return result;
        }

        public static unsafe float[] Transpose(float[] tensorData, int rows, int cols)
        {
            float[] transposedTensorData = new float[tensorData.Length];

            fixed (float* pTensorData = tensorData)
            {
                fixed (float* pTransposedData = transposedTensorData)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        for (int j = 0; j < cols; j++)
                        {
                            int index = i * cols + j;
                            int transposedIndex = j * rows + i;
                            pTransposedData[transposedIndex] = pTensorData[index];
                        }
                    }
                }
            }
            return transposedTensorData;
        }


    }
}
