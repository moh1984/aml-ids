"""Creates tiny fake datasets with the real file layouts, only to check that the code runs.
The numbers produced on these files are meaningless.
    python tests/make_synthetic.py && python tests/smoke.sh
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
F17 = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
       'Total Length of Fwd Packets', 'Fwd Packet Length Max', 'Flow Bytes/s', 'Flow Packets/s',
       'Flow IAT Mean', 'Fwd IAT Total', 'Bwd IAT Total', 'Fwd PSH Flags', 'FIN Flag Count',
       'Average Packet Size', 'Avg Fwd Segment Size', 'Init_Win_bytes_forward',
       'Subflow Fwd Packets', 'Active Mean', 'Idle Mean', 'Packet Length Variance']
F18 = ['Dst Port', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
       'Fwd Pkt Len Max', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Fwd IAT Tot',
       'Bwd IAT Tot', 'Fwd PSH Flags', 'FIN Flag Cnt', 'Pkt Size Avg', 'Fwd Seg Size Avg',
       'Init Fwd Win Byts', 'Subflow Fwd Pkts', 'Active Mean', 'Idle Mean', 'Pkt Len Var']


def block(n, labels, shift):
    y = rng.choice(labels, n, p=None)
    codes = np.array([labels.index(v) for v in y])
    X = rng.gamma(2.0, 1.0, (n, len(F17))) * (1 + codes[:, None] * 0.6 + shift)
    X[:5, 6] = np.inf                                           # CIC files contain inf values
    return X, y


def cic(folder, names, days, prefix):
    os.makedirs(folder, exist_ok=True)
    for i, (day, labs) in enumerate(days.items()):
        X, y = block(900, labs, 0.3 * i)
        df = pd.DataFrame(X, columns=[' ' + c for c in names])    # leading spaces like the real files
        ips = rng.integers(1, 6, (len(y), 2))
        df.insert(0, ' Source IP' if prefix == 17 else 'Src IP', [f'10.0.0.{a}' for a in ips[:, 0]])
        df.insert(1, ' Destination IP' if prefix == 17 else 'Dst IP', [f'10.0.1.{b}' for b in ips[:, 1]])
        df[' Label'] = y
        if prefix == 18:
            df = pd.concat([df.iloc[:400], pd.DataFrame([df.columns], columns=df.columns), df.iloc[400:]])
            df.columns = [c.strip() for c in df.columns]
        df.to_csv(f'{folder}/{day}.csv', index=False)


cic('data/CICIDS2017', F17, {
    'Monday-WorkingHours': ['BENIGN'],
    'Tuesday-WorkingHours': ['BENIGN', 'FTP-Patator', 'SSH-Patator'],
    'Wednesday-workingHours': ['BENIGN', 'DoS Hulk', 'DoS slowloris', 'Heartbleed'],
    'Thursday-WorkingHours-Morning-WebAttacks': ['BENIGN', 'Web Attack \x96 Brute Force', 'Infiltration'],
    'Friday-WorkingHours-Afternoon': ['BENIGN', 'PortScan', 'DDoS', 'Bot']}, 17)
cic('data/CSE-CIC-IDS2018', F18, {
    'Wednesday-14-02-2018_TrafficForML_CICFlowMeter': ['Benign', 'FTP-BruteForce'],
    'Thursday-15-02-2018_TrafficForML_CICFlowMeter': ['Benign', 'DoS attacks-Hulk'],
    'Friday-02-03-2018_TrafficForML_CICFlowMeter': ['Benign', 'Bot', 'DDOS attack-HOIC']}, 18)

os.makedirs('data/NSL-KDD', exist_ok=True)
for split, n in (('KDDTrain+', 1500), ('KDDTest+', 500)):
    labs = ['normal', 'neptune', 'smurf', 'ipsweep', 'guess_passwd', 'buffer_overflow']
    y = rng.choice(labs, n, p=[.5, .2, .1, .1, .07, .03])
    X = rng.gamma(2, 1, (n, 38)) * (1 + np.array([labs.index(v) for v in y])[:, None] * .5)
    df = pd.DataFrame(X)
    df.insert(1, 'p', rng.choice(['tcp', 'udp', 'icmp'], n))
    df.insert(2, 's', rng.choice(['http', 'ftp', 'smtp', 'private'], n))
    df.insert(3, 'f', rng.choice(['SF', 'S0', 'REJ'], n))
    df['label'] = y; df['difficulty'] = 20
    df.to_csv(f'data/NSL-KDD/{split}.txt', header=False, index=False)

os.makedirs('data/UNSW-NB15', exist_ok=True)
for split, n in (('training', 1500), ('testing', 600)):
    cats = ['Normal', 'Exploits', 'Fuzzers', 'Generic', 'Backdoor']
    y = rng.choice(cats, n, p=[.5, .2, .15, .1, .05])
    X = rng.gamma(2, 1, (n, 20)) * (1 + np.array([cats.index(v) for v in y])[:, None] * .5)
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
    df.insert(0, 'id', range(n))
    df['proto'] = rng.choice(['tcp', 'udp'], n); df['service'] = rng.choice(['-', 'http'], n)
    df['state'] = rng.choice(['FIN', 'INT'], n)
    df['attack_cat'] = y; df['label'] = (y != 'Normal').astype(int)
    df.to_csv(f'data/UNSW-NB15/UNSW_NB15_{split}-set.csv', index=False)
print('synthetic data written to ./data')
