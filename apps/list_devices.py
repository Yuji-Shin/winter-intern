import sounddevice as sd

print(sd.query_devices())
print("\nDefault device:", sd.default.device)
