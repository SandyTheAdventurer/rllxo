import PySpice
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

# Create a shared instance to see if Ngspice is detected
try:
    ngspice = NgSpiceShared.new_instance()
    print("Ngspice detected successfully!")
    print("Ngspice version:", ngspice.version)
except Exception as e:
    print("Ngspice detection failed:", e)
