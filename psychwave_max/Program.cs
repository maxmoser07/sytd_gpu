using System;
using System.IO;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Silk.NET.OpenCL;

namespace OpenCLWeb;

class Program
{
    // Global OpenCL Context Variables accessible by the route handler
    private static CL cl = null!;
    private static nint context;
    private static nint commandQueue;
    private static nint kernel;
    private static nint program;
    private static nint deviceBuffer;
    
    private static int width = 800;
    private static int height = 600;
    private static nuint bufferSize;

    static unsafe void Main(string[] args)
    {
        // --- 1. Initialize OpenCL Infrastructure ONCE at Startup ---
        cl = CL.GetApi();
        uint numPlatforms = 0;
        cl.GetPlatformIDs(0, null, &numPlatforms);
        nint* platforms = stackalloc nint[(int)numPlatforms];
        cl.GetPlatformIDs(numPlatforms, platforms, null);

        uint numDevices = 0;
        cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, 0, null, &numDevices);
        nint* devices = stackalloc nint[(int)numDevices];
        cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, numDevices, devices, null);

        int errorCode = 0;
        context = cl.CreateContext(null, 1, devices, null, null, &errorCode);
        commandQueue = cl.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, &errorCode);

        string kernelSource = @"
__kernel void GenerateWaves(__global float* output, const int width, const int height, float time,
                            float zoom, float speed, float distortion, int complexity) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    int index = y * width + x;

    float u = (((float)x / width) * 2.0f - 1.0f) * zoom;
    float v = (((float)y / height) * 2.0f - 1.0f) * zoom;
    float t = time * speed;

    for(int i = 1; i <= complexity; i++) {
        u += sin(v * 1.5f + t + (float)i) * distortion;
        v += cos(u * 1.5f + t - (float)i) * distortion;
    }
    output[index] = sin(u + v) * 0.5f + 0.5f;
}";

        // Using backwards-compatible classic array initialization
        string[] sources = new string[] { kernelSource };
        program = cl.CreateProgramWithSource(context, 1, sources, null, &errorCode);
        cl.BuildProgram(program, 1, devices, (byte*)null, null, null);
        kernel = cl.CreateKernel(program, "GenerateWaves", &errorCode);

        bufferSize = (nuint)(width * height * sizeof(float));
        deviceBuffer = cl.CreateBuffer(context, MemFlags.WriteOnly, bufferSize, null, &errorCode);

        Console.WriteLine("🚀 OpenCL Engine fully loaded on GPU. Starting Web Server...");

        // --- 2. Spin up the Web Server Application ---
        var builder = WebApplication.CreateBuilder(args);
        var app = builder.Build();

        // Appending '; charset=utf-8' forces browsers to correctly render the emojis!
        app.MapGet("/", () => Results.Content(GetHtmlFrontend(), "text/html; charset=utf-8"));

        // Point the route directly to our clean, dedicated static method
        app.MapGet("/render", RenderFrame);

        app.Run("http://localhost:5000");

        // --- 3. Clean up OpenCL handles upon exit ---
        cl.ReleaseMemObject(deviceBuffer);
        cl.ReleaseKernel(kernel);
        cl.ReleaseProgram(program);
        cl.ReleaseCommandQueue(commandQueue);
        cl.ReleaseContext(context);
    }

    // This clean static method allows parameters to sit on the stack frame properly
    private static unsafe IResult RenderFrame(
        float time, float zoom, float speed, float distortion, int complexity,
        double rPh, double gPh, double bPh, double rFr, double gFr, double bFr)
    {
        float[] hostOutput = new float[width * height];

        // Explicitly map local stack copies to protect pointer evaluation from optimization shifts
        float localTime = time;
        float localZoom = zoom;
        float localSpeed = speed;
        float localDistortion = distortion;
        int localComplexity = complexity;
        int localWidth = width;
        int localHeight = height;
        nint localBuffer = deviceBuffer;

        // Safely pass stack-backed variables to OpenCL
        cl.SetKernelArg(kernel, 0, (nuint)sizeof(nint), &localBuffer);
        cl.SetKernelArg(kernel, 1, (nuint)sizeof(int), &localWidth);
        cl.SetKernelArg(kernel, 2, (nuint)sizeof(int), &localHeight);
        cl.SetKernelArg(kernel, 3, (nuint)sizeof(float), &localTime);
        cl.SetKernelArg(kernel, 4, (nuint)sizeof(float), &localZoom);
        cl.SetKernelArg(kernel, 5, (nuint)sizeof(float), &localSpeed);
        cl.SetKernelArg(kernel, 6, (nuint)sizeof(float), &localDistortion);
        cl.SetKernelArg(kernel, 7, (nuint)sizeof(int), &localComplexity);

        nuint[] globalWorkSize = new nuint[] { (nuint)localWidth, (nuint)localHeight };
        fixed (nuint* pGlobalSize = globalWorkSize)
        {
            cl.EnqueueNdrangeKernel(commandQueue, kernel, 2, null, pGlobalSize, null, 0, null, null);
        }

        fixed (float* pHostOutput = hostOutput)
        {
            cl.EnqueueReadBuffer(commandQueue, deviceBuffer, true, (nuint)0, bufferSize, (void*)pHostOutput, 0, null, null);
        }

        // Color palettes
        double[] brightness = new double[] { 0.5, 0.5, 0.5 };
        double[] contrast   = new double[] { 0.5, 0.5, 0.5 };
        double[] frequency  = new double[] { rFr, gFr, bFr };
        double[] phase      = new double[] { rPh, gPh, bPh };

        byte[] bmpBytes = ExportToBmpBytes(hostOutput, localWidth, localHeight, brightness, contrast, frequency, phase);
        
        return Results.File(bmpBytes, "image/bmp");
    }

    private static byte[] ExportToBmpBytes(float[] rawData, int width, int height, double[] brightness, double[] contrast, double[] frequency, double[] phase)
    {
        int rowSize = (width * 3 + 3) & ~3;
        int pixelDataSize = rowSize * height;
        int fileSize = 54 + pixelDataSize;
        byte[] bmpBytes = new byte[fileSize];

        bmpBytes[0] = 0x42; bmpBytes[1] = 0x4D;
        BitConverter.GetBytes(fileSize).CopyTo(bmpBytes, 2);
        BitConverter.GetBytes(54).CopyTo(bmpBytes, 10);
        BitConverter.GetBytes(40).CopyTo(bmpBytes, 14);
        BitConverter.GetBytes(width).CopyTo(bmpBytes, 18);
        BitConverter.GetBytes(height).CopyTo(bmpBytes, 22);
        BitConverter.GetBytes((short)1).CopyTo(bmpBytes, 26);
        BitConverter.GetBytes((short)24).CopyTo(bmpBytes, 28);
        BitConverter.GetBytes(pixelDataSize).CopyTo(bmpBytes, 34);

        double tau = 6.28318530718;
        for (int y = 0; y < height; y++)
        {
            int rowOffset = y * rowSize + 54;
            int dataRowOffset = y * width;
            for (int x = 0; x < width; x++)
            {
                float strength = rawData[dataRowOffset + x];
                byte r = (byte)Math.Clamp((brightness[0] + contrast[0] * Math.Cos(tau * (strength * frequency[0] + phase[0]))) * 255, 0, 255);
                byte g = (byte)Math.Clamp((brightness[1] + contrast[1] * Math.Cos(tau * (strength * frequency[1] + phase[1]))) * 255, 0, 255);
                byte b = (byte)Math.Clamp((brightness[2] + contrast[2] * Math.Cos(tau * (strength * frequency[2] + phase[2]))) * 255, 0, 255);

                int pixelOffset = rowOffset + x * 3;
                bmpBytes[pixelOffset] = b;
                bmpBytes[pixelOffset + 1] = g;
                bmpBytes[pixelOffset + 2] = r;
            }
        }
        return bmpBytes;
    }

    private static string GetHtmlFrontend() => @"
<!DOCTYPE html>
<html>
<head>
    <title>OpenCL PsychWave Dashboard</title>
    <style>
        body { background: #111; color: #eee; font-family: sans-serif; display: flex; padding: 20px; }
        .controls { width: 320px; padding-right: 20px; display: flex; flex-direction: column; gap: 12px; }
        .control-group { background: #222; padding: 10px; border-radius: 6px; }
        label { display: block; font-size: 12px; margin-bottom: 4px; color: #aaa; }
        input[type=range] { width: 100%; }
        span { float: right; font-weight: bold; color: #00ffcc; }
        img { border: 4px solid #333; border-radius: 8px; background: #000; height: 600px; width: 800px; }
    </style>
</head>
<body>
    <div class='controls'>
        <h2>🎛️ Wave Controls</h2>
        <div class='control-group'>
            <label>Time Shift: <span id='v_time'>2.5</span></label>
            <input type='range' id='time' min='0' max='20' step='0.1' value='2.5'>
            <label>Zoom Level: <span id='v_zoom'>3.5</span></label>
            <input type='range' id='zoom' min='0.5' max='10' step='0.1' value='3.5'>
            <label>Warp Distortion: <span id='v_distortion'>0.35</span></label>
            <input type='range' id='distortion' min='0.05' max='1.0' step='0.05' value='0.35'>
            <label>Math Complexity: <span id='v_complexity'>5</span></label>
            <input type='range' id='complexity' min='1' max='8' step='1' value='5'>
        </div>
        <h2>🎨 Color Palette Phases</h2>
        <div class='control-group'>
            <label>Red Phase: <span id='v_rPh'>0.0</span></label><input type='range' id='rPh' min='0' max='1' step='0.05' value='0'>
            <label>Green Phase: <span id='v_gPh'>0.33</span></label><input type='range' id='gPh' min='0' max='1' step='0.05' value='0.33'>
            <label>Blue Phase: <span id='v_bPh'>0.67</span></label><input type='range' id='bPh' min='0' max='1' step='0.05' value='0.67'>
        </div>
        <h2>🌈 Color Frequencies</h2>
        <div class='control-group'>
            <label>Red Frequency: <span id='v_rFr'>1.0</span></label><input type='range' id='rFr' min='0' max='3' step='0.1' value='1'>
            <label>Green Frequency: <span id='v_gFr'>1.0</span></label><input type='range' id='gFr' min='0' max='3' step='0.1' value='1'>
            <label>Blue Frequency: <span id='v_bFr'>1.0</span></label><input type='range' id='bFr' min='0' max='3' step='0.1' value='1'>
        </div>
    </div>
    <div>
        <img id='viewport' src='' />
    </div>

    <script>
        const sliders = ['time', 'zoom', 'distortion', 'complexity', 'rPh', 'gPh', 'bPh', 'rFr', 'gFr', 'bFr'];
        
        function updateViewport() {
            let params = [];
            sliders.forEach(id => {
                const val = document.getElementById(id).value;
                document.getElementById('v_' + id).innerText = val;
                params.push(id + '=' + val);
            });
            document.getElementById('viewport').src = '/render?' + params.join('&') + '&speed=1.0';
        }

        sliders.forEach(id => document.getElementById(id).addEventListener('input', updateViewport));
        updateViewport();
    </script>
</body>
</html>";
}