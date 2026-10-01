"""Dependency-free deterministic reader for classic Ethernet PCAP files."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import struct
from typing import Iterator


class PcapFormatError(ValueError):
    """Raised when a capture cannot be interpreted without guessing."""


@dataclass(frozen=True, slots=True)
class PacketRecord:
    capture_index: int
    timestamp_ns: int
    ip_version: int
    src_addr: str
    dst_addr: str
    protocol_number: int
    src_port: int
    dst_port: int
    ip_total_length: int
    ttl: int
    tcp_flags: int | None = None
    tcp_seq: int | None = None
    tcp_ack: int | None = None
    tcp_window: int | None = None
    payload: bytes = b""


_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000),
    b"\xa1\xb2\xc3\xd4": (">", 1_000),
    b"\x4d\x3c\xb2\xa1": ("<", 1),
    b"\xa1\xb2\x3c\x4d": (">", 1),
}


def _transport(ip: bytes, offset: int, end: int, protocol: int) -> tuple[int, int, dict, bytes]:
    segment = ip[offset:end]
    if protocol == 6:
        if len(segment) < 20:
            raise PcapFormatError("truncated TCP header")
        src, dst, seq, ack = struct.unpack("!HHII", segment[:12])
        header_len = (segment[12] >> 4) * 4
        if header_len < 20 or header_len > len(segment):
            raise PcapFormatError("invalid TCP header length")
        flags = segment[13]
        window = struct.unpack("!H", segment[14:16])[0]
        return src, dst, {"tcp_flags": flags, "tcp_seq": seq,
                          "tcp_ack": ack, "tcp_window": window}, segment[header_len:]
    if protocol == 17:
        if len(segment) < 8:
            raise PcapFormatError("truncated UDP header")
        src, dst, length = struct.unpack("!HHH", segment[:6])
        if length < 8:
            raise PcapFormatError("invalid UDP length")
        return src, dst, {}, segment[8:min(length, len(segment))]
    return 0, 0, {}, segment


def _ethernet_packet(raw: bytes, index: int, timestamp_ns: int) -> PacketRecord | None:
    if len(raw) < 14:
        raise PcapFormatError("truncated Ethernet frame")
    ether_type = struct.unpack("!H", raw[12:14])[0]
    offset = 14
    while ether_type in (0x8100, 0x88A8):
        if len(raw) < offset + 4:
            raise PcapFormatError("truncated VLAN header")
        ether_type = struct.unpack("!H", raw[offset + 2:offset + 4])[0]
        offset += 4
    if ether_type == 0x0800:
        if len(raw) < offset + 20:
            raise PcapFormatError("truncated IPv4 header")
        version_ihl = raw[offset]
        if version_ihl >> 4 != 4:
            raise PcapFormatError("invalid IPv4 version")
        ihl = (version_ihl & 0x0F) * 4
        total = struct.unpack("!H", raw[offset + 2:offset + 4])[0]
        flags_fragment = struct.unpack("!H", raw[offset + 6:offset + 8])[0]
        if ihl < 20 or total < ihl or len(raw) < offset + min(total, ihl):
            raise PcapFormatError("invalid IPv4 lengths")
        if flags_fragment & 0x3FFF:
            raise PcapFormatError("fragmented IPv4 requires explicit reassembly")
        ttl, protocol = raw[offset + 8], raw[offset + 9]
        src = str(ipaddress.IPv4Address(raw[offset + 12:offset + 16]))
        dst = str(ipaddress.IPv4Address(raw[offset + 16:offset + 20]))
        end = min(offset + total, len(raw))
        src_port, dst_port, tcp, payload = _transport(raw, offset + ihl, end, protocol)
        return PacketRecord(index, timestamp_ns, 4, src, dst, protocol, src_port,
                            dst_port, total, ttl, payload=payload, **tcp)
    if ether_type == 0x86DD:
        if len(raw) < offset + 40 or raw[offset] >> 4 != 6:
            raise PcapFormatError("truncated or invalid IPv6 header")
        payload_len = struct.unpack("!H", raw[offset + 4:offset + 6])[0]
        protocol, hop_limit = raw[offset + 6], raw[offset + 7]
        if protocol in {0, 43, 44, 50, 51, 60, 135, 139, 140}:
            raise PcapFormatError("IPv6 extension headers require explicit parsing")
        total = 40 + payload_len
        src = str(ipaddress.IPv6Address(raw[offset + 8:offset + 24]))
        dst = str(ipaddress.IPv6Address(raw[offset + 24:offset + 40]))
        end = min(offset + total, len(raw))
        src_port, dst_port, tcp, payload = _transport(raw, offset + 40, end, protocol)
        return PacketRecord(index, timestamp_ns, 6, src, dst, protocol, src_port,
                            dst_port, total, hop_limit, payload=payload, **tcp)
    return None


def read_pcap(path: str | Path) -> Iterator[PacketRecord]:
    """Yield supported IP packets in capture order; reject ambiguous formats."""
    with Path(path).open("rb") as source:
        magic = source.read(4)
        if magic not in _MAGIC:
            raise PcapFormatError("unsupported PCAP magic (PCAPNG is not silently accepted)")
        endian, fraction_to_ns = _MAGIC[magic]
        rest = source.read(20)
        if len(rest) != 20:
            raise PcapFormatError("truncated PCAP global header")
        _, _, _, _, _, linktype = struct.unpack(endian + "HHIIII", rest)
        if linktype != 1:
            raise PcapFormatError("only DLT_EN10MB Ethernet is supported")
        index = 0
        while True:
            header = source.read(16)
            if not header:
                return
            if len(header) != 16:
                raise PcapFormatError("truncated PCAP packet header")
            seconds, fraction, included, original = struct.unpack(endian + "IIII", header)
            if included > original:
                raise PcapFormatError("included length exceeds original length")
            raw = source.read(included)
            if len(raw) != included:
                raise PcapFormatError("truncated PCAP packet")
            timestamp_ns = seconds * 1_000_000_000 + fraction * fraction_to_ns
            packet = _ethernet_packet(raw, index, timestamp_ns)
            index += 1
            if packet is not None:
                yield packet
