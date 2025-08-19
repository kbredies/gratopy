from __future__ import annotations

import pyopencl as cl


def pytest_report_header():
    print("-----  OpenCL Device information  -----")
    for platform in cl.get_platforms():
        print(f"Platform: {platform.name}")
        for device in platform.get_devices():
            print(f"  Device: {device.name}")
            print(f"    Type: {cl.device_type.to_string(device.type)}")
            print(f"    Version: {device.version}")
            print(f"    Max Compute Units: {device.max_compute_units}")
            print(f"    Max Work Group Size: {device.max_work_group_size}")

    print()
    print("Automatically created OpenCL context:")
    try:
        ctx = cl.create_some_context()
        print(f"  Context: {ctx}")
        for device in ctx.devices:
            print(f"    Device: {device.name}")
            print(f"    Type: {cl.device_type.to_string(device.type)}")
            print(f"    Version: {device.version}")
    except cl.RuntimeError as e:
        print(f"  Error: {e}")
