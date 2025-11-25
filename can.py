import time
import struct
import can


def send_message(frame_ID,value_1,value_2):
    """
    Send a CAN message given an ID in hex (ID 0x100) with 2 signed 32-bit float values.
    DBC layout (little-endian, signed):
      x1: bits 0..31
      y1: bits 32..63
    """

    # Use SocketCAN interface 'can0' (what MCP2518FD usually exposes)
    bus = can.interface.Bus(channel="can0", bustype="socketcan")



    # <hhhh = little-endian, 4 × signed 16-bit ints
    data = struct.pack("<ffff", value_1, value_2)

    msg = can.Message(
        arbitration_id=frame_ID,  # Hex frame ID ex: 0x100
        data=data,
        is_extended_id=False,  # standard ID, matches your BO_ 100
    )

    try:
        frame_ID.send(msg)
        print(
            f"Sent vision: Frame ID={frame_ID}, Vaule_1={value_1}, value_2={value_2}, bytes={data.hex()}"
        )
    except can.CanError as e:
        print(f"Failed to send message: {e}")


def main():
    # Example arbitrary test values
    test_values = [
        (100,100, 200,),
        (100,-100, -200),
        (101,0, 0),
        (101, 0.32767, -0.32768 ),
    ]

    # Send each example value once per second in a loop
    while True:
        for (frame_ID, value_1, value_2) in test_values:
            send_message(frame_ID, value_1, value_2)
            time.sleep(2.0)


if __name__ == "__main__":
    main()


