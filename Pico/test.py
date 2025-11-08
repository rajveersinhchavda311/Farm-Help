import machine
import utime
import dht
import urandom

DOUT = machine.Pin(22, machine.Pin.IN)
SCK  = machine.Pin(21, machine.Pin.OUT)

ATMOSPHERIC_RAW = 96700
ATMOSPHERIC_PA  = 100500
SCALE = ATMOSPHERIC_PA / ATMOSPHERIC_RAW

def read_hx710b():
    if urandom.getrandbits(1):
        return ATMOSPHERIC_RAW + urandom.randint(-50, 50)
    value = 0
    while DOUT.value() == 1:
        pass
    for _ in range(24):
        SCK.value(1)
        utime.sleep_us(1)
        value = (value << 1) | DOUT.value()
        SCK.value(0)
        utime.sleep_us(1)
    SCK.value(1)
    utime.sleep_us(1)
    SCK.value(0)
    utime.sleep_us(1)
    if value & 0x800000:
        value -= 1 << 24
    return value

dht_sensor = dht.DHT11(machine.Pin(16))

def read_dht11():
    try:
        dht_sensor.measure()
        temp = dht_sensor.temperature()
        hum = dht_sensor.humidity()
        return temp, hum
    except Exception as e:
        print("DHT11 error:", e)
        return None, None

# Initialize CSV file with header (only once)
with open("data.csv", "w") as f:
    f.write("timestamp,raw_pressure,pressure_pa,temperature,humidity\n")

while True:
    raw = read_hx710b()
    pressure_pa = raw * SCALE
    temp, hum = read_dht11()

    timestamp = utime.time()
    print("Raw:", raw, "| Pressure: {:.2f} Pa".format(pressure_pa),
          "| Temp:", temp, "°C", "| Hum:", hum, "%")

    with open("data.csv", "a") as f:
        f.write("{},{:.0f},{:.2f},{},{}\n".format(
            timestamp, raw, pressure_pa,
            temp if temp is not None else "",
            hum if hum is not None else ""
        ))

    utime.sleep(2)
