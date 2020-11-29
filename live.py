from scapy.all import sniff
from scapy.all import wrpcap
from scapy.arch.windows import show_interfaces
import os
import livepredict
import pyshark
# cap = pyshark.LiveCapture(interface='ethernet',output_file='data.pcap')
# cap.set_debug()
# cap.sniff(timeout=20)
show_interfaces()
# # pkts = sniff(timeout=15,iface="Npcap Loopback Adapter")
pkts = sniff(timeout=10,iface="Intel(R) Dual Band Wireless-AC 7265")
os.chdir(os.getcwd()+'/livepcap')
wrpcap('data.pcap',pkts)
os.chdir('..')
os.chdir('C:/Users/user/Desktop/prj/cicflowmeter-4/CICFlowMeter-4.0/bin')
os.system("cfm.bat \"C:/Users/user/Desktop/prj/livepcap\" \"C:/Users/user/Desktop/prj/livedata\"")
os.chdir('C:/Users/user/Desktop/prj')
livepredict.predict()