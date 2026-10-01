"""Conservative categorical and application-protocol candidates."""

from __future__ import annotations

import re

from .flow_builder import Flow


PROTOCOL_NAMES = {1: "icmp", 6: "tcp", 17: "udp", 58: "ipv6-icmp"}
HTTP_REQUEST = re.compile(rb"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH|CONNECT|TRACE)\s+")
HTTP_RESPONSE = re.compile(rb"^HTTP/\d(?:\.\d)?\s+")
FTP_COMMAND = re.compile(rb"^(USER|PASS|RETR|STOR|LIST|CWD|PWD|TYPE|PORT|PASV|QUIT)(?:\s|\r|\n)")
FTP_SUCCESS = re.compile(rb"^230(?:\s|-)" )


def protocol_token(flow: Flow) -> str:
    return PROTOCOL_NAMES.get(flow.key.protocol_number, f"ip-{flow.key.protocol_number}")


def application_candidates(flow: Flow) -> dict[str, object]:
    """Return payload-evidenced candidates; never infer a service from a port."""
    http_methods = 0
    http_responses = 0
    response_body_len = 0
    ftp_commands = 0
    ftp_login_success = False
    for packet in flow.packets:
        payload = packet.payload
        if not payload:
            continue
        if HTTP_REQUEST.match(payload):
            http_methods += 1
        if HTTP_RESPONSE.match(payload):
            http_responses += 1
            split = payload.find(b"\r\n\r\n")
            if split >= 0:
                response_body_len += len(payload) - split - 4
        if FTP_COMMAND.match(payload.upper()):
            ftp_commands += 1
        if FTP_SUCCESS.match(payload):
            ftp_login_success = True
    if http_methods or http_responses:
        service = "http"
    elif ftp_commands or ftp_login_success:
        service = "ftp"
    else:
        service = None
    return {
        "service_candidate": service,
        "http_method_count": http_methods,
        "http_transaction_depth_candidate": min(http_methods, http_responses),
        "http_response_body_len_candidate": response_body_len,
        "ftp_command_count_candidate": ftp_commands,
        "ftp_login_success_candidate": True if ftp_login_success else None,
    }


def state_candidate(flow: Flow) -> str | None:
    """A deterministic local TCP summary, explicitly not an Argus mapping."""
    if flow.key.protocol_number != 6:
        return None
    flags = [packet.tcp_flags or 0 for packet in flow.packets]
    if any(value & 0x04 for value in flags):
        return "LOCAL_RST"
    if any(value & 0x01 for value in flags):
        return "LOCAL_FIN"
    syn = any((value & 0x02) and not (value & 0x10) for value in flags)
    syn_ack = any((value & 0x12) == 0x12 for value in flags)
    ack = any((value & 0x10) for value in flags)
    if syn and syn_ack and ack:
        return "LOCAL_ESTABLISHED"
    if syn:
        return "LOCAL_SYN_ONLY"
    return "LOCAL_OTHER"
