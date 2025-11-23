import RPi.GPIO as GPIO
import time

def toggle(pin, sec):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)

    GPIO.output(pin, 1)
    time.sleep(sec)
    GPIO.output(pin, 0)

    GPIO.cleanup()

def toggle_forever(pin, sec):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)

    try:
        while True:
            GPIO.output(pin, 1)
            time.sleep(sec)
            GPIO.output(pin, 0)
            time.sleep(sec)

    except KeyboardInterrupt:
        pass

    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    toggle_forever(17, 2)  # toggles every 2 seconds forever


