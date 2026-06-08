using System;
using System.IO;
using Silk.NET.OpenCL;

namespace OpenCLInit;

class Program
{
    static unsafe void Main(string[] args)
    {
        // 1. Fetch the raw OpenCL API object
        CL cl = CL.GetApi();

        // 2. Discover available Platforms
        uint numPlatforms = 0;
        cl.GetPlatformIDs(0, null, &numPlatforms);

        if (numPlatforms == 0)
        {
            Console.WriteLine("No OpenCL platforms found.");
            return;
        }

        nint* platforms = stackalloc nint[(int)numPlatforms];
        cl.GetPlatformIDs(numPlatforms, platforms, null);

        // Fetch and print the name of the first available platform
        nuint paramValueSize;
        byte[] infoBuffer = new byte[256];
        fixed (byte* pBuffer = infoBuffer)
        {
            cl.GetPlatformInfo(platforms[0], (uint)PlatformInfo.Name, (nuint)infoBuffer.Length, pBuffer, &paramValueSize);
        }
        string platformName = System.Text.Encoding.UTF8.GetString(infoBuffer, 0, (int)paramValueSize).TrimEnd('\0');
        Console.WriteLine($"Platform: {platformName}");

        // 3. Discover GPU Devices
        uint numDevices = 0;
        cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, 0, null, &numDevices);

        if (numDevices == 0)
        {
            Console.WriteLine("No GPU devices found on this platform.");
            return;
        }

        nint* devices = stackalloc nint[(int)numDevices];
        cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, numDevices, devices, null);

        // Fetch and print device name
        fixed (byte* pBuffer = infoBuffer)
        {
            cl.GetDeviceInfo(devices[0], (uint)DeviceInfo.Name, (nuint)infoBuffer.Length, pBuffer, &paramValueSize);
        }
        string deviceName = System.Text.Encoding.UTF8.GetString(infoBuffer, 0, (int)paramValueSize).TrimEnd('\0');
        Console.WriteLine($"Device: {deviceName}");

        // 4. Create an OpenCL Context
        int errorCode = 0;
        nint context = cl.CreateContext(null, 1, devices, null, null, &errorCode);
        if (errorCode != 0)
        {
            Console.WriteLine($"Context creation failed with error code: {errorCode}");
            return;
        }
        Console.WriteLine("Success! OpenCL context successfully initialized on the GPU.");

        // 5. Create a Command Queue
        nint commandQueue = cl.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, &errorCode);
        if (errorCode != 0)
        {
            Console.WriteLine($"Failed to create command queue. Error: {errorCode}");
            cl.ReleaseContext(context);
            return;
        }
        Console.WriteLine("📥 Command queue created.");

        // 6. Psychedelic OpenCL Kernel Source
        string kernelSource = @"
__kernel void GenerateWaves(__global float* output, const int width, const int height, float time) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int index = y * width + x;

    float u = ((float)x / width) * 2.0f - 1.0f;
    float v = ((float)y / height) * 2.0f - 1.0f;

    for(float i = 1.0f; i < 4.0f; i += 1.0f) {
        u += sin(v * 1.5f + time + i) * 0.4f;
        v += cos(u * 1.5f + time - i) * 0.4f;
    }

    output[index] = sin(u + v) * 0.5f + 0.5f;
}
";

        // 7. Create Program
        string[] sources = [kernelSource];
        nint program = cl.CreateProgramWithSource(context, 1, sources, null, &errorCode);
        if (errorCode != 0)
        {
            Console.WriteLine($"Failed to create program. Error: {errorCode}");
            cl.ReleaseCommandQueue(commandQueue);
            cl.ReleaseContext(context);
            return;
        }

        // 8. Compile Program
        errorCode = cl.BuildProgram(program, 1, devices, (byte*)null, null, null);
        if (errorCode != 0)
        {
            Console.WriteLine("❌ GPU Compilation Failed!");
            nuint logSize;
            cl.GetProgramBuildInfo(program, devices[0], (uint)ProgramBuildInfo.BuildLog, 0, null, &logSize);
            byte[] logBuffer = new byte[logSize];
            fixed (byte* pLog = logBuffer)
            {
                cl.GetProgramBuildInfo(program, devices[0], (uint)ProgramBuildInfo.BuildLog, logSize, pLog, null);
            }
            Console.WriteLine(System.Text.Encoding.UTF8.GetString(logBuffer));
            cl.ReleaseProgram(program);
            cl.ReleaseCommandQueue(commandQueue);
            cl.ReleaseContext(context);
            return;
        }
        Console.WriteLine("OpenCL Kernel compiled successfully on the GPU!");

        // 9. Extract Kernel
        nint kernel = cl.CreateKernel(program, "GenerateWaves", &errorCode);
        if (errorCode != 0)
        {
            Console.WriteLine($"Failed to create kernel object. Error: {errorCode}");
            cl.ReleaseProgram(program);
            cl.ReleaseCommandQueue(commandQueue);
            cl.ReleaseContext(context);
            return;
        }
        Console.WriteLine("Kernel 'GenerateWaves' is ready for dispatch.");

        // 10. Dimensions and Buffers
        int width = 800;
        int height = 600;
        int totalPixels = width * height;
        float timeValue = 2.5f; 

        float[] hostOutput = new float[totalPixels];

        // 11. Allocate VRAM Buffer
        nuint bufferSize = (nuint)(totalPixels * sizeof(float));
        nint deviceBuffer = cl.CreateBuffer(context, MemFlags.WriteOnly, bufferSize, null, &errorCode);
        if (errorCode != 0)
        {
            Console.WriteLine($"Failed to allocate VRAM buffer. Error: {errorCode}");
            cl.ReleaseKernel(kernel);
            cl.ReleaseProgram(program);
            cl.ReleaseCommandQueue(commandQueue);
            cl.ReleaseContext(context);
            return;
        }

        // 12. Set Arguments
        cl.SetKernelArg(kernel, 0, (nuint)sizeof(nint), &deviceBuffer);
        cl.SetKernelArg(kernel, 1, (nuint)sizeof(int), &width);
        cl.SetKernelArg(kernel, 2, (nuint)sizeof(int), &height);
        cl.SetKernelArg(kernel, 3, (nuint)sizeof(float), &timeValue);

        // 13. Dispatch Kernel
        nuint[] globalWorkSize = [(nuint)width, (nuint)height];
        fixed (nuint* pGlobalSize = globalWorkSize)
        {
            errorCode = cl.EnqueueNdrangeKernel(
                commandQueue, kernel, 2, (nuint*)null, pGlobalSize, (nuint*)null, 0, (nint*)null, (nint*)null
            );
        }
        
        if (errorCode != 0)
        {
            Console.WriteLine($"Kernel execution failed. Error: {errorCode}");
            cl.ReleaseMemObject(deviceBuffer);
            cl.ReleaseKernel(kernel);
            cl.ReleaseProgram(program);
            cl.ReleaseCommandQueue(commandQueue);
            cl.ReleaseContext(context);
            return;
        }

        // 14. Read back data
        fixed (float* pHostOutput = hostOutput)
        {
            cl.EnqueueReadBuffer(
                commandQueue, deviceBuffer, true, (nuint)0, bufferSize, (void*)pHostOutput, 0, (nint*)null, (nint*)null
            );
        }
        Console.WriteLine("GPU calculation complete! Data successfully read back to CPU.");

        // 15. Export to a 24-bit BMP image without any third-party dependencies!
        string outputPath = "psychedelic_wave.bmp";
        ExportToBmp(outputPath, hostOutput, width, height);

        // Cleanup
        cl.ReleaseMemObject(deviceBuffer);
        cl.ReleaseKernel(kernel);
        cl.ReleaseProgram(program);
        cl.ReleaseCommandQueue(commandQueue);
        cl.ReleaseContext(context);
    }

    private static void ExportToBmp(string path, float[] rawData, int width, int height)
    {
        int rowSize = (width * 3 + 3) & ~3; // Align rows to a 4-byte boundary
        int pixelDataSize = rowSize * height;
        int fileSize = 54 + pixelDataSize;

        byte[] bmpBytes = new byte[fileSize];

        // --- BMP HEADER ---
        bmpBytes[0] = 0x42; bmpBytes[1] = 0x4D;                   // Signature "BM"
        BitConverter.GetBytes(fileSize).CopyTo(bmpBytes, 2);      // File Size
        BitConverter.GetBytes(54).CopyTo(bmpBytes, 10);           // Pixel Data Offset

        // --- DIB HEADER (BITMAPINFOHEADER) ---
        BitConverter.GetBytes(40).CopyTo(bmpBytes, 14);           // Size of Header
        BitConverter.GetBytes(width).CopyTo(bmpBytes, 18);        // Width
        BitConverter.GetBytes(height).CopyTo(bmpBytes, 22);       // Height (Positive means bottom-to-top)
        BitConverter.GetBytes((short)1).CopyTo(bmpBytes, 26);     // Color Planes
        BitConverter.GetBytes((short)24).CopyTo(bmpBytes, 28);    // Bits per pixel (24-bit RGB)
        BitConverter.GetBytes(pixelDataSize).CopyTo(bmpBytes, 34);// Image Size

        // --- RGB PALETTE MAPPING ---
        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * rowSize + 54;
            int dataRowOffset = y * width;

            for (int x = 0; x < width; x++)
            {
                float strength = rawData[dataRowOffset + x];

                // Procedural Cosine Rainbow Mapping matching the original GLSL palette
                // Phase splits (Red = 0.0, Green = 2.09, Blue = 4.18 radians)
                byte r = (byte)((Math.Sin(strength * 6.283185 + 0.00) * 0.5 + 0.5) * 255);
                byte g = (byte)((Math.Sin(strength * 6.283185 + 2.09) * 0.5 + 0.5) * 255);
                byte b = (byte)((Math.Sin(strength * 6.283185 + 4.18) * 0.5 + 0.5) * 255);

                int pixelOffset = rowOffset + x * 3;
                bmpBytes[pixelOffset] = b;     // BMP format expects colors in BGR order
                bmpBytes[pixelOffset + 1] = g;
                bmpBytes[pixelOffset + 2] = r;
            }
        }

        File.WriteAllBytes(path, bmpBytes);
        Console.WriteLine($"visual pattern saved to: {Path.GetFullPath(path)}");
    }
}