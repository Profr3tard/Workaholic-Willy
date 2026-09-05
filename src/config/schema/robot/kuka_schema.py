"""KUKA driver config schema."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .._base import StrictModel


class KukaEkiConfig(StrictModel):
    """EthernetKRL TCP/XML settings for the planned KUKA driver.

    ``role`` is stated from the Python application's side: ``server``, the
    default, means Willy listens and the KRL program opens the connection.
    """

    role: Literal["server", "client"] = Field(default="server")
    host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(default=7000, ge=1, le=65535)
    timeout_s: float = Field(default=5.0, gt=0.0)
    heartbeat_s: float = Field(default=0.5, gt=0.0)
    buffer_size: int = Field(default=65536, ge=1024)


class KukaConfig(StrictModel):
    """Vendor-specific KUKA settings.

    The one transport planned for the first implementation is EthernetKRL
    (EKI/KRL) over TCP/XML.

    ``controller_ip`` is the network address of the KUKA controller itself,
    read when ``eki.role == "client"`` and Willy dials the controller. Under
    ``eki.role == "server"`` the controller dials Willy and ``controller_ip``
    is informational only.
    """

    model: str = Field(default="unconfigured", min_length=1)
    dof: int = Field(default=6, ge=1, le=12)
    controller_ip: str = Field(default="192.168.1.20", min_length=1)
    eki: KukaEkiConfig = Field(default_factory=KukaEkiConfig)
