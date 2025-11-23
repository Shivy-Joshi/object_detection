import RPi.GPIO as GPIO
import time

def toggle(pin, sec):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)

    GPIO.output(pin, 1)
    time.sleep(sec)
    GPIO.output(pin, 0)

    GPIO.cleanup()


if __name__ == "__main__":
    toggle(17, 3000)
# use the GPIO pin not the physical pin number (used 27 for test on LED)

