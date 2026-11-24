"""Explicit aliases, in a fixed statistical-feature order; no identifiers."""
import re
import unicodedata


def normalize(value):
    text = unicodedata.normalize("NFKC", str(value)).replace("\ufeff", "")
    return re.sub(r"[^a-z0-9]", "", text.lower())


# canonical name, CIC-IDS2017 header, CSE-CIC-IDS2018 header.
# Only non-payload flow statistics are used. Header lengths, duplicated/bulk
# columns, windows, ports, timestamps, flow IDs and addresses are excluded.
FEATURE_SPECS = [
    ("flow_duration", "Flow Duration", "Flow Duration"),
    ("fwd_packets", "Total Fwd Packets", "Tot Fwd Pkts"),
    ("bwd_packets", "Total Backward Packets", "Tot Bwd Pkts"),
    ("fwd_bytes", "Total Length of Fwd Packets", "TotLen Fwd Pkts"),
    ("bwd_bytes", "Total Length of Bwd Packets", "TotLen Bwd Pkts"),
    ("fwd_length_min", "Fwd Packet Length Min", "Fwd Pkt Len Min"),
    ("fwd_length_max", "Fwd Packet Length Max", "Fwd Pkt Len Max"),
    ("fwd_length_mean", "Fwd Packet Length Mean", "Fwd Pkt Len Mean"),
    ("fwd_length_std", "Fwd Packet Length Std", "Fwd Pkt Len Std"),
    ("bwd_length_min", "Bwd Packet Length Min", "Bwd Pkt Len Min"),
    ("bwd_length_max", "Bwd Packet Length Max", "Bwd Pkt Len Max"),
    ("bwd_length_mean", "Bwd Packet Length Mean", "Bwd Pkt Len Mean"),
    ("bwd_length_std", "Bwd Packet Length Std", "Bwd Pkt Len Std"),
    ("flow_bytes_s", "Flow Bytes/s", "Flow Byts/s"),
    ("flow_packets_s", "Flow Packets/s", "Flow Pkts/s"),
    ("fwd_packets_s", "Fwd Packets/s", "Fwd Pkts/s"),
    ("bwd_packets_s", "Bwd Packets/s", "Bwd Pkts/s"),
    ("flow_iat_mean", "Flow IAT Mean", "Flow IAT Mean"),
    ("flow_iat_std", "Flow IAT Std", "Flow IAT Std"),
    ("flow_iat_min", "Flow IAT Min", "Flow IAT Min"),
    ("flow_iat_max", "Flow IAT Max", "Flow IAT Max"),
    ("fwd_iat_total", "Fwd IAT Total", "Fwd IAT Tot"),
    ("fwd_iat_mean", "Fwd IAT Mean", "Fwd IAT Mean"),
    ("fwd_iat_std", "Fwd IAT Std", "Fwd IAT Std"),
    ("fwd_iat_min", "Fwd IAT Min", "Fwd IAT Min"),
    ("fwd_iat_max", "Fwd IAT Max", "Fwd IAT Max"),
    ("bwd_iat_total", "Bwd IAT Total", "Bwd IAT Tot"),
    ("bwd_iat_mean", "Bwd IAT Mean", "Bwd IAT Mean"),
    ("bwd_iat_std", "Bwd IAT Std", "Bwd IAT Std"),
    ("bwd_iat_min", "Bwd IAT Min", "Bwd IAT Min"),
    ("bwd_iat_max", "Bwd IAT Max", "Bwd IAT Max"),
    ("packet_length_min", "Min Packet Length", "Pkt Len Min"),
    ("packet_length_max", "Max Packet Length", "Pkt Len Max"),
    ("packet_length_mean", "Packet Length Mean", "Pkt Len Mean"),
    ("packet_length_std", "Packet Length Std", "Pkt Len Std"),
    ("packet_length_variance", "Packet Length Variance", "Pkt Len Var"),
    ("fin_count", "FIN Flag Count", "FIN Flag Cnt"),
    ("syn_count", "SYN Flag Count", "SYN Flag Cnt"),
    ("rst_count", "RST Flag Count", "RST Flag Cnt"),
    ("psh_count", "PSH Flag Count", "PSH Flag Cnt"),
    ("ack_count", "ACK Flag Count", "ACK Flag Cnt"),
    ("urg_count", "URG Flag Count", "URG Flag Cnt"),
    ("ece_count", "ECE Flag Count", "ECE Flag Cnt"),
    ("down_up_ratio", "Down/Up Ratio", "Down/Up Ratio"),
    ("packet_size_average", "Average Packet Size", "Pkt Size Avg"),
    ("fwd_segment_average", "Avg Fwd Segment Size", "Fwd Seg Size Avg"),
    ("bwd_segment_average", "Avg Bwd Segment Size", "Bwd Seg Size Avg"),
    ("active_mean", "Active Mean", "Active Mean"),
    ("active_std", "Active Std", "Active Std"),
    ("active_min", "Active Min", "Active Min"),
    ("active_max", "Active Max", "Active Max"),
    ("idle_mean", "Idle Mean", "Idle Mean"),
    ("idle_std", "Idle Std", "Idle Std"),
    ("idle_min", "Idle Min", "Idle Min"),
    ("idle_max", "Idle Max", "Idle Max"),
]
FEATURES = [spec[0] for spec in FEATURE_SPECS]
ALIASES = {normalize(alias): spec[0] for spec in FEATURE_SPECS for alias in spec}
META_ALIASES = {
    "label": ["Label", "class", "attack_label"],
    "src_ip": ["Source IP", "Src IP", "SourceIP"],
    "dst_ip": ["Destination IP", "Dst IP", "DestinationIP"],
    "src_port": ["Source Port", "Src Port"],
    "dst_port": ["Destination Port", "Dst Port"],
    "protocol": ["Protocol"],
}


def resolve_columns(columns):
    """Return canonical -> original; refuse ambiguous aliases."""
    result = {}
    metadata = {normalize(v): k for k, vs in META_ALIASES.items() for v in vs}
    for col in columns:
        key = ALIASES.get(normalize(col)) or metadata.get(normalize(col))
        if key:
            if key in result:
                raise ValueError(f"Ambiguous columns for {key}: {result[key]!r}, {col!r}")
            result[key] = col
    return result


BENIGN_LABELS = {"benign", "normal", "0"}
ATTACK_LABELS = {
    "attack", "malicious", "1", "ddos", "dosgoldeneye", "doshulk", "dosslowhttptest",
    "dosslowloris", "heartbleed", "ftppatator", "sshpatator", "bot", "botnet",
    "portscan", "infiltration", "infilteration", "webattackbruteforce",
    "webattackxss", "webattacksqlinjection", "ftpbruteforce", "sshbruteforce",
    "dosattacksgoldeneye", "dosattackshulk", "dosattacksslowhttptest",
    "dosattacksslowloris", "ddosattacksloichttp", "ddosattackloichttp",
    "ddosloichttp", "ddosloicudp", "ddoshoic", "ddosattackhoic", "ddosattackloicudp",
    "ddosattackshoic", "ddosattacksloicudp", "bruteforceweb", "bruteforcexss",
    "sqlinjection",
}


def label_value(value):
    key = normalize(value)
    if key in BENIGN_LABELS:
        return 0
    # Some original 2017 CSVs contain non-UTF8 separator bytes in Web Attack
    # labels. Separators are irrelevant; match only documented family endings.
    if key in ATTACK_LABELS or (key.startswith("webattack") and any(
        key.endswith(s) for s in ("bruteforce", "xss", "sqlinjection")
    )):
        return 1
    return None
