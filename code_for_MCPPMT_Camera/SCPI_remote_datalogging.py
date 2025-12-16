from RsInstrument import RsInstrument
import time
# ------------- working code to tell the RTO 1044 to take n_files # with acq_per_file # in data logging mode ---------
'''
1. Install RsInstrument on environment -> 'pip install RsInstrument' 
2. connect via LAN cable to RTO
3. on RTO select IP: 192.168.10.2 and local mask: 255.255.255.0
4. on Mac under Network select manually IP:  192.168.10.1 and local mask: 255.255.255.0 
5. select resolution on the RTO under -> horizontal -> setup 
6. select wanted n_files, acq_per_file, usb_root and wait_s -> typically for 1000 acq's at 1ps resolution: wait_s ~ 30s
7. run script
'''
resource = "TCPIP::192.168.10.2::5025::SOCKET"
scope = RsInstrument(resource,
                     reset=False,
                     id_query=True,
                     options="SelectVisa=socket")

print("Connected to:", scope.query_str("*IDN?"))

scope.write_str("*CLS")

n_files = 3         # number of log files
acq_per_file = 5000   # waveforms per file
usb_root = "D:"
wait_s = 150           # rough time estimate per run (adjust as needed)

for i in range(n_files):
    print(f"Configuring data logging run {i+1}/{n_files}")

    scope.write_str(f"ACQuire:COUNt {acq_per_file}")
    scope.write_str("EXPort:WAVeform:DLOGging ON")
    scope.write_str("EXPort:WAVeform:TIMestamps ON")

    fname = f"{usb_root}\\Run_{i:03d}.bin"
    scope.write_str(f"EXPort:WAVeform:NAME '{fname}'")

    print("Starting RUNSingle")
    scope.write_str("RUNSingle")

    # Just wait long enough for acquisition + logging to finish
    # print(f"Sleeping {wait_s} s while logging...")
    time.sleep(wait_s)

    print(f"Finished run {i+1}, file {fname}")

scope.close()
