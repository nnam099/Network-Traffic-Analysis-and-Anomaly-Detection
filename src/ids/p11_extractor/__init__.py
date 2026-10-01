"""P11 packet-to-semantic extractor research package.

The package is intentionally independent of PyTorch and all P9 model code.
Candidate packet/flow computations are exposed for reference-corpus validation;
they are not claimed to reproduce historical Argus/Bro semantics exactly.
"""

from .flow_builder import Flow, FlowBuilder, FlowKey
from .packet_reader import PacketRecord, read_pcap

__all__ = ["Flow", "FlowBuilder", "FlowKey", "PacketRecord", "read_pcap"]
