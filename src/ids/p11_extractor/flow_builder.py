"""Deterministic candidate bidirectional-flow reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import ip_address
from typing import Iterable

from .packet_reader import PacketRecord


@dataclass(frozen=True, order=True, slots=True)
class Endpoint:
    version: int
    packed: bytes
    port: int

    @classmethod
    def from_packet(cls, version: int, address: str, port: int) -> "Endpoint":
        parsed = ip_address(address)
        if parsed.version != version:
            raise ValueError("packet IP version/address mismatch")
        return cls(version, parsed.packed, int(port))

    @property
    def address(self) -> str:
        return str(ip_address(self.packed))


@dataclass(frozen=True, order=True, slots=True)
class FlowKey:
    protocol_number: int
    endpoint_a: Endpoint
    endpoint_b: Endpoint

    @classmethod
    def from_packet(cls, packet: PacketRecord) -> "FlowKey":
        left = Endpoint.from_packet(packet.ip_version, packet.src_addr, packet.src_port)
        right = Endpoint.from_packet(packet.ip_version, packet.dst_addr, packet.dst_port)
        a, b = sorted((left, right))
        return cls(packet.protocol_number, a, b)


@dataclass(slots=True)
class Flow:
    flow_id: int
    key: FlowKey
    originator: Endpoint
    responder: Endpoint
    packets: list[PacketRecord] = field(default_factory=list)

    @property
    def start_ns(self) -> int:
        return self.packets[0].timestamp_ns

    @property
    def end_ns(self) -> int:
        return self.packets[-1].timestamp_ns

    def is_originator_packet(self, packet: PacketRecord) -> bool:
        endpoint = Endpoint.from_packet(packet.ip_version, packet.src_addr, packet.src_port)
        return endpoint == self.originator


class FlowBuilder:
    """Group a sorted packet stream by bidirectional 5-tuple and idle timeout.

    The timeout is an explicit candidate policy, not a claim about historical
    Argus defaults. Packet ties are ordered by capture index.
    """

    def __init__(self, idle_timeout_ns: int = 60_000_000_000):
        if idle_timeout_ns <= 0:
            raise ValueError("idle_timeout_ns must be positive")
        self.idle_timeout_ns = int(idle_timeout_ns)

    def build(self, packets: Iterable[PacketRecord]) -> list[Flow]:
        ordered = sorted(packets, key=lambda item: (item.timestamp_ns, item.capture_index))
        active: dict[FlowKey, Flow] = {}
        completed: list[Flow] = []
        next_id = 0
        for packet in ordered:
            key = FlowKey.from_packet(packet)
            current = active.get(key)
            if current is not None and packet.timestamp_ns - current.end_ns > self.idle_timeout_ns:
                completed.append(current)
                current = None
            if current is None:
                originator = Endpoint.from_packet(packet.ip_version, packet.src_addr, packet.src_port)
                responder = Endpoint.from_packet(packet.ip_version, packet.dst_addr, packet.dst_port)
                current = Flow(next_id, key, originator, responder)
                next_id += 1
                active[key] = current
            current.packets.append(packet)
        completed.extend(active.values())
        completed.sort(key=lambda flow: (flow.start_ns, flow.flow_id, flow.key))
        return completed
