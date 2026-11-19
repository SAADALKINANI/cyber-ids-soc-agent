"""Curated knowledge base: NSL-KDD attack types -> MITRE ATT&CK techniques -> response playbooks.

The agent may ONLY cite technique IDs that exist in this table; any other ID in an LLM
draft is treated as a hallucination and rejected by the report validator.
"""
from __future__ import annotations

TECHNIQUES = {
    "T1498": "Network Denial of Service",
    "T1499": "Endpoint Denial of Service",
    "T1046": "Network Service Discovery",
    "T1595": "Active Scanning",
    "T1110": "Brute Force",
    "T1190": "Exploit Public-Facing Application",
    "T1068": "Exploitation for Privilege Escalation",
    "T1014": "Rootkit",
    "T1105": "Ingress Tool Transfer",
    "T1071": "Application Layer Protocol",
    "T1572": "Protocol Tunneling",
    "T1059": "Command and Scripting Interpreter",
    "T1048": "Exfiltration Over Alternative Protocol",
    "T1210": "Exploitation of Remote Services",
}

FAMILY_TECHNIQUES = {
    "DoS": ["T1498", "T1499"],
    "Probe": ["T1046", "T1595"],
    "R2L": ["T1110", "T1190", "T1210"],
    "U2R": ["T1068", "T1059"],
    "normal": [],
}

ATTACK_TECHNIQUES = {
    "neptune": ["T1499"], "smurf": ["T1498"], "pod": ["T1499"], "teardrop": ["T1499"],
    "back": ["T1499"], "land": ["T1499"], "apache2": ["T1499"], "mailbomb": ["T1499"],
    "processtable": ["T1499"], "udpstorm": ["T1498"],
    "ipsweep": ["T1595", "T1046"], "nmap": ["T1046"], "portsweep": ["T1046"], "satan": ["T1595", "T1046"],
    "mscan": ["T1595", "T1046"], "saint": ["T1595", "T1046"],
    "guess_passwd": ["T1110"], "ftp_write": ["T1105"], "imap": ["T1210"], "phf": ["T1190"],
    "multihop": ["T1210"], "spy": ["T1048"], "warezclient": ["T1105"], "warezmaster": ["T1105"],
    "httptunnel": ["T1572"], "named": ["T1210"], "sendmail": ["T1210"], "snmpgetattack": ["T1110"],
    "snmpguess": ["T1110"], "xlock": ["T1110"], "xsnoop": ["T1071"], "worm": ["T1210"],
    "buffer_overflow": ["T1068"], "loadmodule": ["T1068"], "perl": ["T1059"], "rootkit": ["T1014"],
    "ps": ["T1068"], "sqlattack": ["T1190"], "xterm": ["T1068"],
}

PLAYBOOKS = {
    "DoS": [
        ("contain", "Apply upstream rate-limiting / scrubbing for the targeted service"),
        ("contain", "Enable SYN cookies and tighten connection timeouts on the target host"),
        ("investigate", "Check whether the source set is spoofed (TTL / geo spread) before blocking"),
    ],
    "Probe": [
        ("contain", "Add the scanning source to a watch-list; block at perimeter if external"),
        ("investigate", "Diff discovered open ports against the approved asset baseline"),
        ("investigate", "Look for follow-on exploitation attempts from the same source in 24h"),
    ],
    "R2L": [
        ("contain", "Lock the targeted account(s) and force credential reset"),
        ("contain", "Isolate the target host from sensitive enclaves pending review"),
        ("investigate", "Pull authentication logs for the account across all systems"),
    ],
    "U2R": [
        ("contain", "Isolate host immediately (EDR network containment)"),
        ("investigate", "Capture memory image before reboot for forensics"),
        ("escalate", "Escalate to Tier-3 / incident commander - possible privilege escalation"),
    ],
    "unknown": [
        ("investigate", "Traffic does not match any known pattern - capture full PCAP for the session"),
        ("escalate", "Route to threat-hunting team for zero-day analysis"),
    ],
}
