import numpy as np
import pandas as pd

df = pd.read_csv("data_full/Monday-WorkingHours.pcap_ISCX.csv",
                 encoding="latin1", nrows=300000, low_memory=False)
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Flow Bytes/s"])


def close(a, b):
    return np.isclose(a.astype(float), b.astype(float), rtol=1e-3, atol=1e-2)


def pct(mask):
    return f"{100 * mask.mean():.1f}%"


fwd, bwd = df["Total Fwd Packets"], df["Total Backward Packets"]
n = fwd + bwd
tot = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
dur = df["Flow Duration"]
m2 = n == 2
f2 = fwd == 2

print("rows:", len(df))
print("01 AvgPktSize == tot/n        :", pct(close(df["Average Packet Size"], tot / n)))
print("02 AvgPktSize == tot/(n-1)    :", pct(close(df["Average Packet Size"][n > 1], (tot / (n - 1))[n > 1])))
print("03 PktLenMean == tot/n        :", pct(close(df["Packet Length Mean"], tot / n)))
print("04 PktLenMean == tot/(n+1)    :", pct(close(df["Packet Length Mean"], tot / (n + 1))))
print("05 Fwd header per pkt (top)   :", (df["Fwd Header Length"] / fwd).round(1).value_counts().head(6).to_dict())
print("06 min_seg_size_fwd TCP (top) :", df.loc[df["Protocol"] == 6, "min_seg_size_forward"].value_counts().head(6).to_dict())
print("07 min_seg_size_fwd UDP (top) :", df.loc[df["Protocol"] == 17, "min_seg_size_forward"].value_counts().head(6).to_dict())
print("08 DownUp == floor(bwd/fwd)   :", pct(close(df["Down/Up Ratio"], np.floor(bwd / fwd))))
print("09 DownUp == bwd/fwd          :", pct(close(df["Down/Up Ratio"], bwd / fwd)))
print("10 InitWinFwd -1 share TCP/UDP:", pct(df.loc[df["Protocol"] == 6, "Init_Win_bytes_forward"] == -1),
      "/", pct(df.loc[df["Protocol"] == 17, "Init_Win_bytes_forward"] == -1))
print("11 InitWinBwd -1 when bwd==0  :", pct(df.loc[bwd == 0, "Init_Win_bytes_backward"] == -1))
print("12 n==2: FlowIATMean == dur   :", pct(close(df.loc[m2, "Flow IAT Mean"], dur[m2])))
print("13 n==2: FlowIATMax == dur    :", pct(close(df.loc[m2, "Flow IAT Max"], dur[m2])))
d = (df["Fwd Packet Length Max"] - df["Fwd Packet Length Min"]).abs()
print("14 fwd==2: std sample (n-1)   :", pct(close(df.loc[f2, "Fwd Packet Length Std"], (d / np.sqrt(2))[f2])))
print("15 fwd==2: std population (n) :", pct(close(df.loc[f2, "Fwd Packet Length Std"], (d / 2)[f2])))
print("16 PktLenVar == PktLenStd^2   :", pct(close(df["Packet Length Variance"], df["Packet Length Std"] ** 2)))
print("17 bwd==0: FwdIATTot == dur   :", pct(close(df.loc[bwd == 0, "Fwd IAT Total"], dur[bwd == 0])))
print("18 FlowBytes/s == tot/(dur/1e6):", pct(close(df.loc[dur > 0, "Flow Bytes/s"], (tot / (dur / 1e6))[dur > 0])))
print("19 FlowPkts/s == n/(dur/1e6)  :", pct(close(df.loc[dur > 0, "Flow Packets/s"], (n / (dur / 1e6))[dur > 0])))
print("20 AvgFwdSeg == FwdLenMean    :", pct(close(df["Avg Fwd Segment Size"], df["Fwd Packet Length Mean"])))
print("21 FwdLenMean == totfwd/fwd   :", pct(close(df["Fwd Packet Length Mean"], df["Total Length of Fwd Packets"] / fwd)))
mn = np.where(bwd > 0, np.minimum(df["Fwd Packet Length Min"], df["Bwd Packet Length Min"]), df["Fwd Packet Length Min"])
print("22 MinPktLen == min(fwd,bwd)  :", pct(close(df["Min Packet Length"], pd.Series(mn, index=df.index))))
print("23 IdleMean>0 when dur>5s     :", pct(df.loc[dur > 5e6, "Idle Mean"] > 0))
print("24 ActiveMean>0 when dur>5s   :", pct(df.loc[dur > 5e6, "Active Mean"] > 0))
print("25 Fwd PSH Flags values       :", df["Fwd PSH Flags"].value_counts().head(4).to_dict())
print("26 Bulk cols all zero         :", bool((df[["Fwd Avg Bytes/Bulk", "Bwd Avg Bulk Rate"]] == 0).all().all()))
print("27 FIN Flag Count values      :", df["FIN Flag Count"].value_counts().head(4).to_dict())
print("28 dur max (sec)              :", dur.max() / 1e6)