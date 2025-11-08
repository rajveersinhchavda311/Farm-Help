import machine
import utime
import dht

# DHT11 connected to GPIO15 (change if needed)
sensor = dht.DHT11(machine.Pin(16))

while True:
    try:
        sensor.measure()  # Trigger measurement
        temp = sensor.temperature()   # °C
        hum = sensor.humidity()       # %
        
        print("Temperature: {}°C  |  Humidity: {}%".format(temp, hum))
        
    except Exception as e:
        print("Sensor error:", e)
    
    utime.sleep(2)  # wait 2 seconds
