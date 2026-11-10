import numpy as np
import pandas as pd

df = pd.read_csv("data_full/Monday-WorkingHours.pcap_ISCX.csv",
                 encoding="latin1", nrows=300000, low_memory=False)
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Flow Bytes/s"])


def close(a, b):
    return np.isclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-3, atol=1e-2)


def pct(mask):
    return f"{100 * np.mean(mask):.1f}%"


fwd, bwd = df["Total Fwd Packets"], df["Total Backward Packets"]
n = fwd + bwd
tot = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
mean, avg = df["Packet Length Mean"], df["Average Packet Size"]
nz = tot > 0

# --- 1) first packet counted twice in Packet Length stats?
print("A1 Avg*n == Mean*(n+1)          :", pct(close((avg * n)[nz], (mean * (n + 1))[nz])))
p1 = mean * (n + 1) - tot
ok = (p1 >= df["Fwd Packet Length Min"] - 0.5) & (p1 <= df["Fwd Packet Length Max"] + 0.5)
print("A2 extra value inside fwd range :", pct(ok[nz]))
one = nz & (fwd == 1)
print("A3 fwd==1: extra == FwdMax      :", pct(close(p1[one], df["Fwd Packet Length Max"][one])))
s = (fwd == 1) & (bwd == 1) & nz
f, b = df["Fwd Packet Length Max"][s], df["Bwd Packet Length Max"][s]
std3 = np.std(np.vstack([f, f, b]), axis=0, ddof=1)
std2 = np.std(np.vstack([f, b]), axis=0, ddof=1)
print("A4 1+1 flows: std of [f,f,b]    :", pct(close(df["Packet Length Std"][s], std3)))
print("A5 1+1 flows: std of [f,b]      :", pct(close(df["Packet Length Std"][s], std2)))
print("A6 MaxPktLen == max(fwd,bwd)    :", pct(close(df["Max Packet Length"], np.maximum(df["Fwd Packet Length Max"], df["Bwd Packet Length Max"]))))

# --- 2) header length
tcp, udp = df["Protocol"] == 6, df["Protocol"] == 17
print("B1 UDP fwd header per pkt (top) :", (df["Fwd Header Length"] / fwd)[udp].round(1).value_counts().head(5).to_dict())
print("B2 UDP bwd header per pkt (top) :", (df["Bwd Header Length"] / bwd)[udp & (bwd > 0)].round(1).value_counts().head(5).to_dict())
print("B3 HeaderLen == HeaderLen.1     :", pct(df["Fwd Header Length"] == df["Fwd Header Length.1"]))
print("B4 TCP fwd==1: minseg==FwdHdr   :", pct((df["min_seg_size_forward"] == df["Fwd Header Length"])[tcp & (fwd == 1)]))
print("B5 UDP fwd==1: minseg==FwdHdr   :", pct((df["min_seg_size_forward"] == df["Fwd Header Length"])[udp & (fwd == 1)]))
print("B6 UDP fwd==1: FwdHdr values    :", df.loc[udp & (fwd == 1), "Fwd Header Length"].value_counts().head(4).to_dict())
print("B7 UDP fwd==1: minseg values    :", df.loc[udp & (fwd == 1), "min_seg_size_forward"].value_counts().head(4).to_dict())

# --- 3) flags: counts or 0/1 ?
big = tcp & (n >= 6)
for c in ["FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
          "ACK Flag Count", "URG Flag Count", "ECE Flag Count", "CWE Flag Count",
          "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags"]:
    print(f"C  {c:<16} (TCP, n>=6) max={df.loc[big, c].max()}  values={df.loc[big, c].value_counts().head(3).to_dict()}")
print("C9 PSH Count == Fwd PSH Flags   :", pct((df["PSH Flag Count"] == df["Fwd PSH Flags"])[tcp]))

# --- 4) misc
print("D1 Subflow FwdPkts == TotFwd    :", pct(df["Subflow Fwd Packets"] == fwd))
print("D2 Subflow FwdBytes == TotLenFwd:", pct(df["Subflow Fwd Bytes"] == df["Total Length of Fwd Packets"]))
print("D3 act_data_pkt_fwd <= fwd      :", pct(df["act_data_pkt_fwd"] <= fwd))
print("D4 IdleMin >= 5s when Idle>0    :", pct(df.loc[df["Idle Mean"] > 0, "Idle Min"] >= 5e6))
print("D5 AvgBwdSeg == BwdLenMean      :", pct(close(df["Avg Bwd Segment Size"], df["Bwd Packet Length Mean"])))
print("D6 sample row (fwd=2,bwd=2):")
print(df.loc[(fwd == 2) & (bwd == 2) & tcp & nz,
             ["Fwd Packet Length Max", "Fwd Packet Length Min", "Bwd Packet Length Max", "Bwd Packet Length Min",
              "Packet Length Mean", "Average Packet Size", "Packet Length Std", "Min Packet Length"]].head(3).to_string())