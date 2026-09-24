"""Explicit classic-PCAP/Ethernet parser candidate for P13.

This parser is synthetically validated but remains unactivated until a
reserved source's capture-format audit demonstrates that it is applicable.
Fragmented packets and truncated captures are excluded with counters; no
implicit library reassembly or default flow behavior is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import struct

from ids.p12_portable_schema import PacketObservation


PARSER_VERSION = "p13-classic-pcap-ethernet-v1"
DLT_EN10MB = 1


class CaptureFormatError(ValueError):
    """Raised when the file-level capture format is unsupported or corrupt."""


@dataclass(frozen=True, slots=True)
class ParseResult:
    packets: tuple[PacketObservation, ...]
    audit: dict[str, object]


_MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", "microseconds", 1_000),
    b"\xa1\xb2\xc3\xd4": (">", "microseconds", 1_000),
    b"\x4d\x3c\xb2\xa1": ("<", "nanoseconds", 1),
    b"\xa1\xb2\x3c\x4d": (">", "nanoseconds", 1),
}


def _ports_and_flags(segment: bytes, protocol: int) -> tuple[int, int, int | None]:
    if protocol == 6:
        if len(segment) < 20:
            raise ValueError("truncated TCP header")
        src, dst = struct.unpack("!HH", segment[:4])
        header_length = (segment[12] >> 4) * 4
        if header_length < 20 or header_length > len(segment):
            raise ValueError("invalid TCP header length")
        return src, dst, segment[13]
    if protocol == 17:
        if len(segment) < 8:
            raise ValueError("truncated UDP header")
        src, dst, length = struct.unpack("!HHH", segment[:6])
        if length < 8:
            raise ValueError("invalid UDP length")
        return src, dst, None
    return 0, 0, None


def _ethernet_ip(
    raw: bytes, capture_index: int, timestamp_ns: int, audit: dict[str, object]
) -> PacketObservation | None:
    if len(raw) < 14:
        raise ValueError("truncated Ethernet header")
    ether_type = struct.unpack("!H", raw[12:14])[0]
    offset = 14
    vlan_depth = 0
    while ether_type in {0x8100, 0x88A8}:
        if len(raw) < offset + 4:
            raise ValueError("truncated VLAN header")
        ether_type = struct.unpack("!H", raw[offset + 2:offset + 4])[0]
        offset += 4
        vlan_depth += 1
        if vlan_depth > 2:
            audit["unsupported_encapsulation_packets"] += 1
            return None
    if vlan_depth:
        audit["vlan_packets"] += 1
        audit["maximum_vlan_depth"] = max(int(audit["maximum_vlan_depth"]), vlan_depth)
    if ether_type == 0x0800:
        if len(raw) < offset + 20:
            raise ValueError("truncated IPv4 header")
        first = raw[offset]
        if first >> 4 != 4:
            raise ValueError("invalid IPv4 version")
        ihl = (first & 0x0F) * 4
        total_length = struct.unpack("!H", raw[offset + 2:offset + 4])[0]
        if ihl < 20 or total_length < ihl or len(raw) < offset + total_length:
            raise ValueError("invalid or truncated IPv4 packet length")
        flags_fragment = struct.unpack("!H", raw[offset + 6:offset + 8])[0]
        if flags_fragment & 0x3FFF:
            audit["ipv4_fragment_packets_excluded"] += 1
            return None
        protocol = raw[offset + 9]
        src = str(ipaddress.IPv4Address(raw[offset + 12:offset + 16]))
        dst = str(ipaddress.IPv4Address(raw[offset + 16:offset + 20]))
        src_port, dst_port, tcp_flags = _ports_and_flags(
            raw[offset + ihl:offset + total_length], protocol
        )
        audit["ipv4_packets_emitted"] += 1
        return PacketObservation(
            capture_index, timestamp_ns, 4, src, dst, protocol, src_port, dst_port,
            total_length, tcp_flags,
        )
    if ether_type == 0x86DD:
        if len(raw) < offset + 40 or raw[offset] >> 4 != 6:
            raise ValueError("truncated or invalid IPv6 header")
        payload_length = struct.unpack("!H", raw[offset + 4:offset + 6])[0]
        next_header = raw[offset + 6]
        total_length = 40 + payload_length
        if len(raw) < offset + total_length:
            raise ValueError("truncated IPv6 packet length")
        if next_header == 44:
            audit["ipv6_fragment_packets_excluded"] += 1
            return None
        if next_header in {0, 43, 50, 51, 60, 135, 139, 140}:
            audit["unsupported_ipv6_extension_packets"] += 1
            return None
        src = str(ipaddress.IPv6Address(raw[offset + 8:offset + 24]))
        dst = str(ipaddress.IPv6Address(raw[offset + 24:offset + 40]))
        src_port, dst_port, tcp_flags = _ports_and_flags(
            raw[offset + 40:offset + total_length], next_header
        )
        audit["ipv6_packets_emitted"] += 1
        return PacketObservation(
            capture_index, timestamp_ns, 6, src, dst, next_header, src_port, dst_port,
            total_length, tcp_flags,
        )
    audit["non_ip_packets_excluded"] += 1
    return None


def read_classic_pcap(path: str | Path) -> ParseResult:
    """Read a classic PCAP deterministically and return packets plus audit counts."""
    packets: list[PacketObservation] = []
    path = Path(path)
    with path.open("rb") as source:
        magic = source.read(4)
        if magic not in _MAGIC:
            raise CaptureFormatError("unsupported capture magic; PCAPNG is not accepted implicitly")
        endian, precision, fraction_to_ns = _MAGIC[magic]
        global_header = source.read(20)
        if len(global_header) != 20:
            raise CaptureFormatError("truncated PCAP global header")
        major, minor, _zone, _sigfigs, snaplen, link_type = struct.unpack(
            endian + "HHIIII", global_header
        )
        if (major, minor) != (2, 4):
            raise CaptureFormatError("only classic PCAP version 2.4 is supported")
        if link_type != DLT_EN10MB:
            raise CaptureFormatError("only DLT_EN10MB Ethernet is supported by this parser version")
        audit: dict[str, object] = {
            "parser_version": PARSER_VERSION,
            "capture_format": "classic_pcap",
            "pcap_version": "2.4",
            "byte_order": "little" if endian == "<" else "big",
            "timestamp_precision": precision,
            "timestamp_normalization": "integer nanoseconds since Unix epoch",
            "snaplen": snaplen,
            "link_type": link_type,
            "link_type_name": "DLT_EN10MB",
            "packet_records": 0,
            "packets_emitted": 0,
            "malformed_packets_excluded": 0,
            "truncated_capture_packets_excluded": 0,
            "non_ip_packets_excluded": 0,
            "vlan_packets": 0,
            "maximum_vlan_depth": 0,
            "ipv4_packets_emitted": 0,
            "ipv6_packets_emitted": 0,
            "ipv4_fragment_packets_excluded": 0,
            "ipv6_fragment_packets_excluded": 0,
            "unsupported_ipv6_extension_packets": 0,
            "unsupported_encapsulation_packets": 0,
            "fragment_policy": "exclude every IPv4/IPv6 fragment; no reassembly; count explicitly",
            "malformed_policy": "exclude malformed packet record and count; file-level corruption is fatal",
            "unsupported_policy": "exclude and count; never reinterpret",
        }
        index = 0
        while True:
            packet_header = source.read(16)
            if not packet_header:
                break
            if len(packet_header) != 16:
                raise CaptureFormatError("truncated PCAP packet-record header")
            seconds, fraction, included, original = struct.unpack(endian + "IIII", packet_header)
            raw = source.read(included)
            if len(raw) != included:
                raise CaptureFormatError("truncated PCAP packet-record payload")
            audit["packet_records"] += 1
            if included > original or included > snaplen:
                audit["malformed_packets_excluded"] += 1
                index += 1
                continue
            if included < original:
                audit["truncated_capture_packets_excluded"] += 1
                index += 1
                continue
            timestamp_ns = seconds * 1_000_000_000 + fraction * fraction_to_ns
            try:
                packet = _ethernet_ip(raw, index, timestamp_ns, audit)
            except (ValueError, struct.error):
                audit["malformed_packets_excluded"] += 1
                packet = None
            if packet is not None:
                packets.append(packet)
            index += 1
        audit["packets_emitted"] = len(packets)
        audit["first_timestamp_ns"] = packets[0].timestamp_ns if packets else None
        audit["last_timestamp_ns"] = packets[-1].timestamp_ns if packets else None
        audit["capture_duration_ns"] = (
            packets[-1].timestamp_ns - packets[0].timestamp_ns if packets else None
        )
    return ParseResult(tuple(packets), audit)
