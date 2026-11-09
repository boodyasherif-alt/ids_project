\# ML-based Intrusion Detection System (CICIDS2017)



A machine-learning IDS that classifies network flows as benign or one of several DoS/DDoS attacks, shows the results in a Streamlit dashboard, and sends alert notifications that include the attacker IP.



> \*\*Note:\*\* the demo replays recorded flows from the CICIDS2017 dataset. It is not live traffic capture.



\## Pipeline



flows (with Source/Destination IP) -> XGBoost classifier -> alert -> Discord webhook notification



\## Dataset



CICIDS2017 (Canadian Institute for Cybersecurity, UNB), `GeneratedLabelledFlows` CSVs.

Files used: Wednesday (DoS Hulk, GoldenEye, slowloris, Slowhttptest) and Friday afternoon (DDoS).

The dataset is not included in this repository. Download it from the official page:

https://www.unb.ca/cic/datasets/ids-2017.html



\## Evaluation (three setups, results differ a lot)



| Setup | Accuracy | Note |

|---|---|---|

| Random split on Wednesday | 0.9993 | Optimistic: train and test come from the same sessions |

| Train Wednesday, test Friday DDoS | 0.7848 | Unseen attack type, attack recall 62% |

| Per-class temporal split (Wed + Fri) | 0.9520 | Closest to realistic use |



Model comparison on the temporal split:



| Model | Accuracy | Macro F1 |

|---|---|---|

| Random Forest | 0.9537 | 0.9172 |

| XGBoost | 0.9520 | 0.9436 |



\## Limitations



\- Only DoS/DDoS attacks are covered (no brute force, port scan, web attacks).

\- XGBoost raises more false alarms on benign traffic (about 12k benign flows labeled DDoS in the test split).

\- The source IP comes from recorded data. In practice an IP does not identify a person (VPN, NAT, compromised hosts).

\- Results on one dataset do not guarantee performance on real networks.



\## Run



Set the notification webhook in the terminal before running (never commit it):



\## Files



\- `app.py` - Streamlit dashboard and replay

\- `ids\_cic\_final.ipynb` - training and evaluation

\- `xgb\_model.joblib`, `label\_encoder.joblib`, `feature\_cols.joblib` - trained model files



