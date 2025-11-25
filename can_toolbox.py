import time
import struct
import can

def setup_can0():
    """Configure can0 like your shell commands do."""
    cmds = [
        # ignore errors when bringing it down (same as '|| true' and redirect)
        ["ip", "link", "set", "can0", "down"],
        # set type + bitrate
        [
            "ip", "link", "set", "can0",
            "type", "can",
            "bitrate", "500000",
            "sample-point", "0.875",
            "berr-reporting", "on",
        ],
        # bring it up
        ["ip", "link", "set", "can0", "up"],
    ]

    # 1) down: ignore failure, hide output
    subprocess.run(cmds[0],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL,
                   check=False)

    # 2) set type/bitrate (fail if this breaks)
    subprocess.run(cmds[1], check=True)

    # 3) up (fail if this breaks)
    subprocess.run(cmds[2], check=True)


def open_bus():
    # call this AFTER setup_can0()
    return can.Bus(interface="socketcan", channel="can0")
# Create the bus ONCE globally
bus = can.interface.Bus(channel="can0", bustype="socketcan")


def send_message(frame_ID, value_1, value_2):
    """
    Send a CAN message with 2 float32 values.
    """

    # Pack two floats = 8 bytes
    data = struct.pack("<ff", value_1, value_2)

    msg = can.Message(
        arbitration_id=frame_ID,
        data=data,
        is_extended_id=False,
    )

    try:
        bus.send(msg)
        print(
            f"Sent: ID=0x{frame_ID:X}, v1={value_1}, v2={value_2}, bytes={data.hex()}"
        )
    except can.CanError as e:
        print(f"Failed to send message: {e}")


def main():
    tests = [
        (0x100, 100, 200),
        (0x100, -100, -200),
        (0x101, 0, 0),
        (0x101, 0.32767, -0.32768),
    ]

    while True:
        for frame_ID, v1, v2 in tests:
            send_message(frame_ID, v1, v2)
            time.sleep(2.0)


if __name__ == "__main__":
    main()
