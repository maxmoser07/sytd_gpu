using System;
using System.IO;
using System.Text;
using Silk.NET.OpenCL;

// Define image dimensions
const int Width = 1200;
const int Height = 800;
const string OutputPath = "/home/flo/RiderProjects/SYTD/SYTD_GPU/Image_Generator/psychedelic_wave.bmp";

// Load the OpenCL implementation
CL cl = CL.GetApi();

// -------- Step 1: Initialize Platform & Device ------------------------------------
uint numPlatforms;
unsafe { cl.GetPlatformIDs(0u, (nint*)null, out numPlatforms); }

if (numPlatforms == 0)
{
    Console.WriteLine("No OpenCL platforms found.");
    return;
}

nint[] platforms = new nint[numPlatforms];
unsafe { fixed (nint* p = platforms) cl.GetPlatformIDs(numPlatforms, p, (uint*)null); }

// Select the first platform and find its devices
nint platform = platforms[0];
uint numDevices;
unsafe { cl.GetDeviceIDs(platform, DeviceType.All, 0u, (nint*)null, out numDevices); }

nint[] devices = new nint[numDevices];
unsafe { fixed (nint* p = devices) cl.GetDeviceIDs(platform, DeviceType.All, numDevices, p, (uint*)null); }

nint device = devices[0]; // Choose the primary processing device
Console.WriteLine($"Using Device: {DeviceString(cl, device, DeviceInfo.Name)}");

// -------- Step 2: Create Context & Command Queue ----------------------------------
int err;
nint context;
// FIX: Use 'default' instead of 'null' for the struct-based PfnContextCallback
unsafe { context = cl.CreateContext((nint*)null, 1u, &device, default, (void*)null, &err); }
CheckError(err, "Failed to create OpenCL context");

nint queue;
unsafe { queue = cl.CreateCommandQueue(context, device, (CommandQueueProperties)0, &err); }CheckError(err, "Failed to create command queue");

// -------- Step 3: Compile the OpenCL Wave Kernel ---------------------------------
string kernelSource = @"
__kernel void GenerateWave(__global uchar4* output, int width, int height, float time) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;

    float2 fragCoord = (float2)((float)x, (float)y);
    float2 res = (float2)((float)width, (float)height);
    float2 p = (fragCoord - 0.5f * res) / res.y;
    
    float t = time * 0.4f;

    float2 warp = (float2)(
        sin(p.x * 4.0f + p.y * 3.0f + t),
        cos(p.y * 4.0f - p.x * 2.0f + t * 1.2f)
    );
    p += warp * 0.5f;

    float wave1 = sin(p.x * 5.0f + t);
    float wave2 = cos(p.y * 6.0f - t * 1.5f);
    float wave3 = sin(length(p) * 8.0f - t * 2.0f);
    
    float value = (wave1 + wave2 + wave3) / 3.0f;
    value = value * 3.0f + t * 0.2f;
    value = value - floor(value);

    float3 bias      = (float3)(0.5f, 0.5f, 0.5f);
    float3 amplitude = (float3)(0.5f, 0.5f, 0.5f);
    float3 frequency = (float3)(1.0f, 1.0f, 1.0f);
    float3 phase     = (float3)(0.0f, 0.33f, 0.67f);

    float3 color = bias + amplitude * cos(6.283185f * (frequency * value + phase));

    uchar4 pixel;
    pixel.x = (uchar)(color.z * 255.0f); // Blue
    pixel.y = (uchar)(color.y * 255.0f); // Green
    pixel.z = (uchar)(color.x * 255.0f); // Red
    pixel.w = 255;                       // Alpha

    output[y * width + x] = pixel;
}";

nint program;
string[] sources = [ kernelSource ];

// FIX: When using the managed string[] array overload, we must use the managed 'out err'.
// We also pass a managed 'null' for lengths so OpenCL auto-calculates string length.
unsafe
{
    program = cl.CreateProgramWithSource(context, 1u, sources, null, out err);
}
CheckError(err, "Failed to create program from source");

// FIX: Use 'default' for the struct-based PfnProgramBuildCallback
unsafe { err = cl.BuildProgram(program, 1u, &device, (byte*)null, default, (void*)null); }
if (err != 0)
{
    nuint logSize;
    unsafe { cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, 0u, (void*)null, out logSize); }
    byte[] logBuf = new byte[(int)logSize];
    unsafe { fixed (byte* p = logBuf) cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, logSize, p, out _); }
    Console.WriteLine("OpenCL Compilation Error:\n" + Encoding.UTF8.GetString(logBuf));
    return;
}

nint kernel;
unsafe
{
    byte[] nameBytes = Encoding.UTF8.GetBytes("GenerateWave\0");
    fixed (byte* namePtr = nameBytes)
        kernel = cl.CreateKernel(program, namePtr, &err);
}
CheckError(err, "Failed to create OpenCL kernel handle");

// -------- Step 4: Allocate Hardware and Host Memory ------------------------------
nuint bufferSize = (nuint)(Width * Height * 4); // 4 Bytes per pixel (BGRA)
nint gpuBuffer;
unsafe { gpuBuffer = cl.CreateBuffer(context, MemFlags.WriteOnly, bufferSize, (void*)null, &err); }
CheckError(err, "Failed to allocate VRAM buffer");

// -------- Step 5: Execute the Simulation -----------------------------------------
float timeValue = 5.5f; 
int widthParam = Width;
int heightParam = Height;

unsafe
{
    cl.SetKernelArg(kernel, 0u, (nuint)sizeof(nint), &gpuBuffer);
    cl.SetKernelArg(kernel, 1u, (nuint)sizeof(int), &widthParam);
    cl.SetKernelArg(kernel, 2u, (nuint)sizeof(int), &heightParam);
    cl.SetKernelArg(kernel, 3u, (nuint)sizeof(float), &timeValue);
}

nuint[] globalWorkSize = [(nuint)Width, (nuint)Height];
unsafe
{
    fixed (nuint* gws = globalWorkSize)
    {
        err = cl.EnqueueNdrangeKernel(queue, kernel, 2u, (nuint*)null, gws, (nuint*)null, 0u, (nint*)null, (nint*)null);
    }
}
CheckError(err, "Failed to queue execution grid processing parameters");

// Ensure GPU work completes
cl.Finish(queue);

// -------- Step 6: Extract Buffer and Save -----------------------------------------
byte[] hostImageBytes = new byte[Width * Height * 4];
unsafe
{
    fixed (byte* hostPtr = hostImageBytes)
    {
        // FIX: true for blocking read, 0 for nuint offset, and (void*) cast for the host pointer
        err = cl.EnqueueReadBuffer(queue, gpuBuffer, true, 0, bufferSize, (void*)hostPtr, 0u, (nint*)null, (nint*)null);
    }
}
CheckError(err, "Failed to transfer generated matrix back to host system memory");
cl.Finish(queue);

// Export using custom BMP writer
SaveBgraArrayToBmp(hostImageBytes, Width, Height, OutputPath);
Console.WriteLine($"Successfully generated and exported artifact to '{OutputPath}'!");

// -------- Step 7: Clean Up Native Allocations --------------------------------------
cl.ReleaseMemObject(gpuBuffer);
cl.ReleaseKernel(kernel);
cl.ReleaseProgram(program);
cl.ReleaseCommandQueue(queue);
cl.ReleaseContext(context);


// ==========================================================================================
// Helper Subroutines
// ==========================================================================================

static void CheckError(int err, string operation)
{
    if (err != 0)
    {
        throw new Exception($"{operation} failed with OpenCL error code: {err}");
    }
}

static string DeviceString(CL cl, nint device, DeviceInfo info)
{
    nuint size;
    unsafe { cl.GetDeviceInfo(device, info, 0u, (void*)null, out size); }
    byte[] buf = new byte[(int)size];
    unsafe { fixed (byte* p = buf) cl.GetDeviceInfo(device, info, size, p, out _); }
    return Encoding.UTF8.GetString(buf).TrimEnd('\0');
}

static void SaveBgraArrayToBmp(byte[] pixelData, int width, int height, string filePath)
{
    using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
    using var writer = new BinaryWriter(stream);

    uint fileHeaderSize = 14;
    uint infoHeaderSize = 40;
    uint fileSize = fileHeaderSize + infoHeaderSize + (uint)pixelData.Length;

    writer.Write((byte)'B');
    writer.Write((byte)'M');
    writer.Write(fileSize);
    writer.Write((ushort)0); 
    writer.Write((ushort)0); 
    writer.Write(fileHeaderSize + infoHeaderSize); 

    writer.Write(infoHeaderSize);
    writer.Write(width);
    writer.Write(-height); // Forces top-down coordinate orientation
    writer.Write((ushort)1);  
    writer.Write((ushort)32); 
    writer.Write(0u);         
    writer.Write((uint)pixelData.Length);
    writer.Write(0); 
    writer.Write(0); 
    writer.Write(0u); 
    writer.Write(0u); 

    writer.Write(pixelData);
}