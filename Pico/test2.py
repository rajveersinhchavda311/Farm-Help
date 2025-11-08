import machine
import utime

# HX710B pins
DOUT = machine.Pin(22, machine.Pin.IN)   # OUT/DT → GP22
SCK  = machine.Pin(21, machine.Pin.OUT)  # SCK → GP21

# Calibration constants (use values suggested from your samples + map pressure 1005 hPa)
ATMOSPHERIC_RAW =96700  # raw value at current atmospheric pressure
ATMOSPHERIC_PA  = 100500   # Pa at ~220m elevation
SCALE = ATMOSPHERIC_PA / ATMOSPHERIC_RAW  # Pa per raw unit

def read_hx710b(timeout_ms=500):
    # Wait for data ready (DOUT goes low)
    start = utime.ticks_ms()
    while DOUT.value() == 1:
        if utime.ticks_diff(utime.ticks_ms(), start) > timeout_ms:
            return None
    value = 0
    for _ in range(24):
        SCK.value(1)
        utime.sleep_us(5)
        value = (value << 1) | DOUT.value()
        SCK.value(0)
        utime.sleep_us(5)
    # extra pulse for channel/gain
    SCK.value(1); utime.sleep_us(5); SCK.value(0); utime.sleep_us(5)

    # signed 24-bit
    if value & 0x800000:
        value -= 1 << 24
    return value

# Main
while True:
    raw = read_hx710b()
    if raw is None:
        print("HX710B not responding (timeout).")
    else:
        pressure_pa = raw * SCALE
        pressure_hpa = pressure_pa / 100.0
        print("Raw:", raw, " Pressure: {:.2f} Pa  ({:.2f} hPa)".format(pressure_pa, pressure_hpa))
    utime.sleep(2)
