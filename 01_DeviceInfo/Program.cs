// ============================================================
// Sample 01 – Device Info
// ============================================================
// Goal: Understand the OpenCL hardware model before writing any code.
//
// OpenCL concepts introduced:
//   Platform – A vendor's OpenCL runtime (e.g. Apple, AMD, NVIDIA, Intel).
//   Device   – A physical processor: CPU, GPU, or accelerator.
//   Handle   – An opaque ID that identifies a C object. Stored as 'nint'.
//
// This sample performs no computation. It only queries and prints
// properties of all OpenCL devices available on this machine.
// ============================================================

using System.Text;
using Silk.NET.OpenCL;

// CL.GetApi() loads the OpenCL shared library (e.g. OpenCL.dll / libOpenCL.so)
// and returns a managed wrapper object with one method per OpenCL function.
CL cl = CL.GetApi();

// -------- Step 1: Count available platforms ----------------------------------------
//
// OpenCL uses a two-call pattern for retrieving lists:
//   First call  – pass 0 for count and null for the array → get the actual count.
//   Second call – pass the count and a pre-allocated array → fill the array.
//
// 'unsafe' enables raw pointer types (nint*, uint*) which the C API requires.
uint numPlatforms;
unsafe { cl.GetPlatformIDs(0u, (nint*)null, out numPlatforms); }
//                              ^           ^
//                          null pointer   count returned here

if (numPlatforms == 0)
{
    Console.WriteLine("No OpenCL platform found. Install a GPU driver or a CPU runtime (e.g. Intel OpenCL).");
    return;
}

// -------- Step 2: Retrieve platform handles ----------------------------------------
// 'nint' (native-sized integer) stores an opaque handle – a number that identifies
// a C object. We never interpret the bits; we just pass it back to OpenCL functions.
nint[] platforms = new nint[numPlatforms];
unsafe
{
    // fixed {} pins the managed array in RAM so the garbage collector (GC) cannot
    // relocate it while the native OpenCL function is reading through our pointer.
    // Inside the block we have a raw nint* pointer to the array's first element.
    fixed (nint* p = platforms)
        cl.GetPlatformIDs(numPlatforms, p, (uint*)null);
}

Console.WriteLine($"Found {numPlatforms} OpenCL platform(s):\n");
Console.WriteLine(new string('=', 62));

foreach (nint platform in platforms)
{
    Console.WriteLine($"Platform : {PlatformString(cl, platform, PlatformInfo.Name)}");
    Console.WriteLine($"  Vendor : {PlatformString(cl, platform, PlatformInfo.Vendor)}");
    Console.WriteLine($"  Version: {PlatformString(cl, platform, PlatformInfo.Version)}");

    // -------- Step 3: Count and list devices on this platform ----------------------
    uint numDevices;
    unsafe { cl.GetDeviceIDs(platform, DeviceType.All, 0u, (nint*)null, out numDevices); }

    if (numDevices == 0)
    {
        Console.WriteLine("  (no devices found)\n");
        continue;
    }

    nint[] devices = new nint[numDevices];
    unsafe
    {
        fixed (nint* p = devices)
            cl.GetDeviceIDs(platform, DeviceType.All, numDevices, p, (uint*)null);
    }

    Console.WriteLine($"  Devices: {numDevices}");
    Console.WriteLine(new string('-', 62));

    foreach (nint device in devices)
    {
        // -------- Step 4: Query device properties ----------------------------------
        // GetDeviceInfo returns raw bytes. For scalar values the generic overload
        // (T0& param_value) lets us pass a ref to a typed variable – no unsafe needed.

        uint  computeUnits = DeviceInfoU32(cl, device, DeviceInfo.MaxComputeUnits);
        uint  clockMHz     = DeviceInfoU32(cl, device, DeviceInfo.MaxClockFrequency);
        ulong globalMemB   = DeviceInfoU64(cl, device, DeviceInfo.GlobalMemSize);
        ulong localMemB    = DeviceInfoU64(cl, device, DeviceInfo.LocalMemSize);
        nuint maxWGSize    = DeviceInfoNuint(cl, device, DeviceInfo.MaxWorkGroupSize);

        // The device type is a bit-field (CPU=2, GPU=4, Accelerator=8).
        ulong typeBits = DeviceInfoU64(cl, device, DeviceInfo.Type);
        string typeStr = typeBits switch
        {
            2 => "CPU",
            4 => "GPU",
            8 => "Accelerator",
            _ => $"Other (0x{typeBits:X})"
        };

        Console.WriteLine($"  Name             : {DeviceString(cl, device, DeviceInfo.Name)}");
        Console.WriteLine($"  Type             : {typeStr}");
        Console.WriteLine($"  Compute Units    : {computeUnits}");
        Console.WriteLine($"  Max Clock        : {clockMHz} MHz");
        Console.WriteLine($"  Global Memory    : {globalMemB / 1_048_576} MB");
        Console.WriteLine($"  Local Memory     : {localMemB / 1_024} KB");
        Console.WriteLine($"  Max Work-Group   : {maxWGSize} work-items");
        Console.WriteLine($"  Driver Version   : {DeviceString(cl, device, DeviceInfo.DriverVersion)}");
        Console.WriteLine($"  Device Version   : {DeviceString(cl, device, DeviceInfo.Version)}");
        Console.WriteLine();
    }
}

// ==========================================================================================
// Helper functions
// ==========================================================================================

// Read a null-terminated UTF-8 string via clGetPlatformInfo.
static string PlatformString(CL cl, nint platform, PlatformInfo info)
{
    nuint size;
    unsafe { cl.GetPlatformInfo(platform, info, 0u, (void*)null, out size); }
    byte[] buf = new byte[(int)size];
    unsafe { fixed (byte* p = buf) cl.GetPlatformInfo(platform, info, size, p, out _); }
    return Encoding.UTF8.GetString(buf).TrimEnd('\0');
}

// Read a null-terminated UTF-8 string via clGetDeviceInfo.
static string DeviceString(CL cl, nint device, DeviceInfo info)
{
    nuint size;
    unsafe { cl.GetDeviceInfo(device, info, 0u, (void*)null, out size); }
    byte[] buf = new byte[(int)size];
    unsafe { fixed (byte* p = buf) cl.GetDeviceInfo(device, info, size, p, out _); }
    return Encoding.UTF8.GetString(buf).TrimEnd('\0');
}

// Read a 32-bit unsigned integer via clGetDeviceInfo.
static uint DeviceInfoU32(CL cl, nint device, DeviceInfo info)
{
    uint v = 0;
    // In unsafe code we use a pointer (&v) instead of 'ref'. This mirrors the C API exactly.
    unsafe { cl.GetDeviceInfo(device, info, (nuint)sizeof(uint), &v, (nuint*)null); }
    return v;
}

// Read a 64-bit unsigned integer via clGetDeviceInfo.
static ulong DeviceInfoU64(CL cl, nint device, DeviceInfo info)
{
    ulong v = 0;
    unsafe { cl.GetDeviceInfo(device, info, (nuint)sizeof(ulong), &v, (nuint*)null); }
    return v;
}

// Read a native-sized unsigned integer (= size_t in C) via clGetDeviceInfo.
static nuint DeviceInfoNuint(CL cl, nint device, DeviceInfo info)
{
    nuint v = 0;
    // sizeof(nuint) == IntPtr.Size == 8 on 64-bit, 4 on 32-bit.
    unsafe { cl.GetDeviceInfo(device, info, (nuint)sizeof(nuint), &v, (nuint*)null); }
    return v;
}
