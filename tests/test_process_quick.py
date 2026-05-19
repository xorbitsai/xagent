#!/usr/bin/env python3
"""Quick test of ProcessService functionality."""

import asyncio


async def test_process_service():
    """Test ProcessService basic functionality."""
    from xagent.core.execution.service import ProcessService

    print("Creating ProcessService...")
    service = ProcessService(n_workers=2, address="localhost:12399")

    print("Starting ProcessService...")
    await service.start()

    try:
        print(f"✅ Service status: {service.status.value}")

        # Test health check
        is_healthy = await service.health_check()
        print(f"✅ Health check: {is_healthy}")

        # Get service info
        info = service.get_info()
        print(f"✅ Service info: {info.to_dict()}")

        # Test Python execution
        print("\nTesting Python execution...")
        result = await service.execute_python(
            code="print('Hello from isolated process!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')",
            timeout=10,
        )

        if result.success:
            print("✅ Python execution successful!")
            print(f"Output:\n{result.output}")
        else:
            print(f"❌ Python execution failed: {result.error}")

        # Test command execution
        print("\nTesting command execution...")
        result = await service.execute_command(
            command="echo 'Hello from command!'",
            timeout=10,
        )

        if result.success:
            print("✅ Command execution successful!")
            print(f"Output: {result.output}")
        else:
            print(f"❌ Command execution failed: {result.error}")

    finally:
        print("\nStopping ProcessService...")
        await service.stop()
        print(f"✅ Service status: {service.status.value}")


if __name__ == "__main__":
    asyncio.run(test_process_service())
    print("\n✅ All tests passed!")
