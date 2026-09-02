"""Memory diagnostics for Bountiful infrastructure.

Provides snapshot_memory() for logging memory usage at key lifecycle
events: server startup, pre-fold, post-fold, and transitions between
sequential server modes.
"""

import logging

import psutil

from basic_bot.config import EMBEDDING_PORT, SUMMARY_PORT, CHAT_PORT

logger = logging.getLogger(__name__)

_PORT_ROLES = {
    EMBEDDING_PORT: "embedding",
    SUMMARY_PORT: "summary",
    CHAT_PORT: "chat",
}


def _find_llama_servers() -> dict[str, dict]:
    """Find running llama-server processes and their memory usage.

    Returns a dict keyed by role with pid and rss_mb.
    Identifies role by parsing --port from the process command line.
    """
    servers = {}
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            name = proc.info["name"] or ""
            if "llama-server" not in name:
                continue

            cmdline = proc.info.get("cmdline") or []
            role = "unknown"
            for i, arg in enumerate(cmdline):
                if arg == "--port" and i + 1 < len(cmdline):
                    try:
                        port = int(cmdline[i + 1])
                        role = _PORT_ROLES.get(port, f"port-{port}")
                    except ValueError:
                        pass
                    break

            rss = proc.info["memory_info"].rss
            servers[role] = {
                "pid": proc.info["pid"],
                "rss_mb": round(rss / (1024 * 1024), 1),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return servers


def snapshot_memory(label: str) -> dict:
    """Capture and log memory state at a labeled moment.

    Call at key lifecycle events:
        snapshot_memory("pre-fold")
        snapshot_memory("chat-stopped")
        snapshot_memory("embedding-done")
        snapshot_memory("summary-done")
        snapshot_memory("chat-resumed")

    Returns a dict with the full snapshot for programmatic use.
    """
    servers = _find_llama_servers()

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    snapshot = {
        "label": label,
        "servers": servers,
        "system_total_mb": round(vm.total / (1024 * 1024), 1),
        "system_available_mb": round(vm.available / (1024 * 1024), 1),
        "system_used_percent": vm.percent,
        "swap_used_mb": round(swap.used / (1024 * 1024), 1),
    }

    # Build log line
    server_parts = []
    for role in ("embedding", "summary", "chat"):
        if role in servers:
            server_parts.append(f"{role}={servers[role]['rss_mb']:.0f}MB")

    # Include any unexpected servers
    for role, info in servers.items():
        if role not in ("embedding", "summary", "chat"):
            server_parts.append(f"{role}={info['rss_mb']:.0f}MB")

    servers_str = ", ".join(server_parts) if server_parts else "none"
    system_str = (
        f"{snapshot['system_available_mb']:.0f}/"
        f"{snapshot['system_total_mb']:.0f}MB available "
        f"({snapshot['system_used_percent']}% used)"
    )
    swap_str = f"swap={snapshot['swap_used_mb']:.0f}MB"

    logger.info(
        "Memory [%s]: %s | system %s | %s",
        label, servers_str, system_str, swap_str,
    )

    return snapshot
