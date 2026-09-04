"""Resolve private SWE task images to pinned linux/amd64 OCI manifests.

This is a dataset-independent provenance stage shared by R2E-Gym, SWE-Gym,
SWE-ReBench, and future repository environments.  It never reads dotenv files
or prompt rows.  Registry 404 is the only condition recorded as ``missing``;
authentication, rate-limit, transport, and malformed-response failures abort
without producing a resolved task.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCK_SCHEMA = "miles-oci-image-lock-v1"
TASK_SCHEMA = "miles-swe-task-v1"
IMAGE_PUBLISHER_POLICY = "miles-swe-image-publisher-policy-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,447}@sha256:[0-9a-f]{64}")
_REGISTRY_COMPONENT = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_REGISTRY_HOST = re.compile(r"[a-z0-9][a-z0-9.-]*(?::[0-9]{1,5})?")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_R2E_REPOSITORIES = frozenset(
    {
        "namanjain12/aiohttp_final",
        "namanjain12/coveragepy_final",
        "namanjain12/datalad_final",
        "namanjain12/numpy_final",
        "namanjain12/orange3_final",
        "namanjain12/pandas_final",
        "namanjain12/pillow_final",
        "namanjain12/pyramid_final",
        "namanjain12/scrapy_final",
        "namanjain12/sympy_final",
        "namanjain12/tornado_final",
    }
)
_R2E_DATASETS = frozenset(
    {
        "R2E-Gym/R2E-Gym-Subset",
        "R2E-Gym/R2E-Gym-V1",
    }
)
_SWE_GYM_REPOSITORY = re.compile(
    r"xingyaoww/sweb\.eval\.x86_64\.[a-z0-9][a-z0-9._-]{0,240}"
)
_SWEBENCH_VERIFIED_REPOSITORY = re.compile(
    r"swebench/sweb\.eval\.x86_64\.[a-z0-9][a-z0-9._-]{0,240}"
)
_SWEBENCH_VERIFIED_DATASET = "princeton-nlp/SWE-bench_Verified"
_SWE_REBENCH_REPOSITORY = re.compile(
    r"swerebenchv2/[a-z0-9][a-z0-9._-]{0,240}"
)
_SWE_REBENCH_DATASETS = frozenset(
    {
        "PrimeIntellect/SWE-rebench-V2-Filtered-Verified",
        "nebius/SWE-rebench-V2",
    }
)
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_INDEX_MEDIA_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
_MANIFEST_MEDIA_TYPES = {
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
}
_CONFIG_MEDIA_TYPES = {
    "application/vnd.oci.image.config.v1+json",
    "application/vnd.docker.container.image.v1+json",
}
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_TOKEN_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}
_CHALLENGE_CACHE: dict[tuple[str, str], str] = {}
_TOKEN_CACHE_LOCK = threading.Lock()


class ImageNotFoundError(RuntimeError):
    """The registry authoritatively returned 404 for an image manifest."""


@dataclass(frozen=True)
class _PrivateFileFingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


@dataclass(frozen=True)
class ImageLockConfig:
    """Paths and selection for one generic private-manifest lock run."""

    private_manifest: Path
    locked_manifest: Path
    image_lock_manifest: Path
    resolve_missing: bool = False
    refresh_missing: bool = False
    instance_id: str | None = None
    limit: int | None = None
    concurrency: int = 8
    checkpoint_batch_size: int = 32


def parse_image_reference(image: str) -> tuple[str, str, str, str]:
    """Return registry, display registry, repository, and tag/digest reference."""
    if "/" not in image:
        raise ValueError(f"SWE image is not an explicit repository reference: {image!r}")
    if "@" in image:
        name, reference = image.rsplit("@", maxsplit=1)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", reference) is None:
            raise ValueError(f"SWE image has an invalid digest: {image!r}")
    else:
        last_slash = image.rfind("/")
        last_colon = image.rfind(":")
        if last_colon <= last_slash:
            raise ValueError(f"SWE image has no explicit tag: {image!r}")
        name, reference = image[:last_colon], image[last_colon + 1 :]
        if _TAG.fullmatch(reference) is None:
            raise ValueError(f"SWE image has an invalid tag: {image!r}")
    components = name.split("/")
    first = components[0].lower()
    if "." in first or ":" in first or first == "localhost":
        registry = (
            "registry-1.docker.io"
            if first in {"docker.io", "index.docker.io", "registry-1.docker.io"}
            else first
        )
        repository_parts = components[1:]
        display_registry = "docker.io" if registry == "registry-1.docker.io" else first
    else:
        registry = "registry-1.docker.io"
        repository_parts = components
        display_registry = "docker.io"
    if len(repository_parts) == 1 and registry == "registry-1.docker.io":
        repository_parts.insert(0, "library")
    if (
        _REGISTRY_HOST.fullmatch(registry) is None
        or not repository_parts
        or any(_REGISTRY_COMPONENT.fullmatch(part) is None for part in repository_parts)
    ):
        raise ValueError(f"SWE image has an invalid registry/repository: {image!r}")
    return registry, display_registry, "/".join(repository_parts), reference


def parse_tagged_image(image: str) -> tuple[str, str, str, str]:
    """Parse an explicitly tagged image, rejecting already-pinned references."""
    if "@" in image:
        raise ValueError(f"SWE image is already digest-pinned: {image!r}")
    return parse_image_reference(image)


def validate_task_image_policy(row: Mapping[str, Any]) -> None:
    """Require the exact dataset identity and publisher namespace for each env."""

    source_schema = _required_text(row, "source_schema")
    source_dataset = _required_text(row, "source_dataset")
    sandbox = _required_mapping(row, "sandbox")
    if sandbox.get("backend_selector") != "harbor":
        raise ValueError("SWE task image policy requires the Harbor backend")
    registry, _, repository, _ = parse_image_reference(
        _required_text(sandbox, "source_image")
    )
    if registry != "registry-1.docker.io":
        raise ValueError("SWE task image publisher policy requires Docker Hub")
    source_metadata = _required_mapping(row, "source_metadata")
    split = source_metadata.get("split")
    if source_schema == "swebench":
        if (
            source_dataset != _SWEBENCH_VERIFIED_DATASET
            or split != "test"
            or row.get("eval_only") is not True
            or _SWEBENCH_VERIFIED_REPOSITORY.fullmatch(repository) is None
        ):
            raise ValueError(
                "SWE-bench Verified image is outside its eval-only publisher policy"
            )
        return
    if split != "train" or row.get("eval_only") is not False:
        raise ValueError("SWE task image publisher policy accepts only training rows")
    if source_schema == "r2e-gym-v1":
        if source_dataset not in _R2E_DATASETS or repository not in _R2E_REPOSITORIES:
            raise ValueError("R2E task image is outside its trusted publisher policy")
        return
    if source_schema == "swe-gym":
        if (
            source_dataset != "SWE-Gym/SWE-Gym"
            or _SWE_GYM_REPOSITORY.fullmatch(repository) is None
        ):
            raise ValueError("SWE-Gym image is outside its trusted publisher policy")
        return
    if source_schema == "swe-rebench-v2":
        if (
            source_dataset not in _SWE_REBENCH_DATASETS
            or _SWE_REBENCH_REPOSITORY.fullmatch(repository) is None
        ):
            raise ValueError("SWE-ReBench image is outside its trusted publisher policy")
        if source_dataset.startswith("PrimeIntellect/") and (
            source_metadata.get("image_reference_transform")
            != "prime-filtered-to-upstream-dockerhub-v1"
            or not isinstance(source_metadata.get("published_source_image"), str)
            or not source_metadata["published_source_image"].startswith(
                "prime/primeintellect/"
            )
        ):
            raise ValueError("filtered ReBench image canonicalization provenance is invalid")
        return
    raise ValueError(f"SWE source schema has no image publisher policy: {source_schema}")


def _read_bounded_response(response: Any, *, limit: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                raise RuntimeError("registry response exceeds the admission size limit")
        except ValueError as exc:
            raise RuntimeError("registry returned an invalid Content-Length") from exc
    body = response.read(limit + 1)
    if len(body) > limit:
        raise RuntimeError("registry response exceeds the admission size limit")
    return body


def _bearer_parameters(challenge: str) -> dict[str, str]:
    if not challenge.lower().startswith("bearer "):
        raise RuntimeError("registry returned an unsupported authentication challenge")
    parameters = dict(re.findall(r'([A-Za-z][A-Za-z0-9_-]*)="([^"]*)"', challenge[7:]))
    realm = parameters.get("realm")
    if not realm:
        raise RuntimeError("registry bearer challenge has no realm")
    parsed = urllib.parse.urlparse(realm)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("registry bearer realm must be an authenticated HTTPS endpoint")
    return parameters


def _request_bearer_token(
    challenge: str,
    *,
    registry: str,
    force_refresh: bool = False,
) -> str:
    parameters = _bearer_parameters(challenge)
    query = {key: parameters[key] for key in ("service", "scope") if parameters.get(key)}
    token_url = parameters["realm"]
    separator = "&" if urllib.parse.urlparse(token_url).query else "?"
    if query:
        token_url += separator + urllib.parse.urlencode(query)
    headers = {"Accept": "application/json", "User-Agent": "miles-swe-image-lock/1"}
    parsed_realm = urllib.parse.urlparse(parameters["realm"])
    username = os.environ.get("DOCKERHUB_USERNAME", "")
    access_token = os.environ.get("DOCKERHUB_TOKEN", "")
    if bool(username) != bool(access_token):
        raise RuntimeError("Docker Hub credentials must set both process-environment fields")
    cache_key = (registry, challenge, username)
    with _TOKEN_CACHE_LOCK:
        cached = _TOKEN_CACHE.get(cache_key)
        if not force_refresh and cached is not None and cached[1] > time.monotonic():
            return cached[0]
    if (
        username
        and registry == "registry-1.docker.io"
        and parsed_realm.hostname == "auth.docker.io"
    ):
        encoded = base64.b64encode(f"{username}:{access_token}".encode()).decode("ascii")
        headers["Authorization"] = f"Basic {encoded}"
    request = urllib.request.Request(token_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = _read_bounded_response(response, limit=1024 * 1024)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("registry bearer-token request failed") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("registry bearer-token response is invalid JSON") from exc
    token = (value.get("token") or value.get("access_token")) if isinstance(value, dict) else None
    if not isinstance(token, str) or not token:
        raise RuntimeError("registry bearer-token response contains no token")
    expires_in = value.get("expires_in", 300)
    if isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)):
        raise RuntimeError("registry bearer-token response has an invalid expiry")
    lifetime = max(1.0, min(float(expires_in), 3600.0) - 30.0)
    with _TOKEN_CACHE_LOCK:
        _TOKEN_CACHE[cache_key] = (token, time.monotonic() + lifetime)
    return token


def _manifest_url(registry: str, repository: str, reference: str) -> str:
    quoted_repository = "/".join(
        urllib.parse.quote(part, safe="") for part in repository.split("/")
    )
    quoted_reference = urllib.parse.quote(reference, safe=":")
    return f"https://{registry}/v2/{quoted_repository}/manifests/{quoted_reference}"


def get_manifest(
    registry: str,
    repository: str,
    reference: str,
    *,
    token: str | None = None,
    authentication_attempts: int = 0,
) -> tuple[bytes, str]:
    """Fetch and content-verify one manifest, following a bearer challenge once."""
    if token is None and authentication_attempts == 0:
        with _TOKEN_CACHE_LOCK:
            cached_challenge = _CHALLENGE_CACHE.get((registry, repository))
        if cached_challenge is not None:
            token = _request_bearer_token(cached_challenge, registry=registry)
            authentication_attempts = 1
    headers = {"Accept": _MANIFEST_ACCEPT, "User-Agent": "miles-swe-image-lock/1"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        _manifest_url(registry, repository, reference),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = _read_bounded_response(response, limit=_MAX_MANIFEST_BYTES)
            digest = response.headers.get("Docker-Content-Digest")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ImageNotFoundError(
                f"registry returned 404 for {repository}:{reference}"
            ) from exc
        if exc.code == 401 and authentication_attempts < 2:
            challenge = exc.headers.get("WWW-Authenticate")
            if not challenge:
                raise RuntimeError("registry returned 401 without a bearer challenge") from exc
            bearer_token = _request_bearer_token(
                challenge,
                registry=registry,
                force_refresh=token is not None,
            )
            with _TOKEN_CACHE_LOCK:
                _CHALLENGE_CACHE[(registry, repository)] = challenge
            return get_manifest(
                registry,
                repository,
                reference,
                token=bearer_token,
                authentication_attempts=authentication_attempts + 1,
            )
        raise RuntimeError(f"registry manifest request failed with HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("registry manifest request failed before an authoritative response") from exc
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise RuntimeError("registry manifest response has no valid content digest")
    if "sha256:" + hashlib.sha256(body).hexdigest() != digest:
        raise RuntimeError("registry manifest body does not match Docker-Content-Digest")
    return body, digest


def _blob_url(registry: str, repository: str, digest: str) -> str:
    quoted_repository = "/".join(
        urllib.parse.quote(part, safe="") for part in repository.split("/")
    )
    return f"https://{registry}/v2/{quoted_repository}/blobs/{urllib.parse.quote(digest, safe=':')}"


def _get_blob(
    registry: str,
    repository: str,
    digest: str,
    *,
    token: str | None = None,
    authentication_attempts: int = 0,
) -> bytes:
    if token is None and authentication_attempts == 0:
        with _TOKEN_CACHE_LOCK:
            cached_challenge = _CHALLENGE_CACHE.get((registry, repository))
        if cached_challenge is not None:
            token = _request_bearer_token(cached_challenge, registry=registry)
            authentication_attempts = 1
    headers = {"Accept": "application/octet-stream", "User-Agent": "miles-swe-image-lock/1"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(_blob_url(registry, repository, digest), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = _read_bounded_response(response, limit=_MAX_CONFIG_BYTES)
    except urllib.error.HTTPError as exc:
        if exc.code == 401 and authentication_attempts < 2:
            challenge = exc.headers.get("WWW-Authenticate")
            if not challenge:
                raise RuntimeError("registry returned 401 without a bearer challenge") from exc
            bearer_token = _request_bearer_token(
                challenge,
                registry=registry,
                force_refresh=token is not None,
            )
            with _TOKEN_CACHE_LOCK:
                _CHALLENGE_CACHE[(registry, repository)] = challenge
            return _get_blob(
                registry,
                repository,
                digest,
                token=bearer_token,
                authentication_attempts=authentication_attempts + 1,
            )
        raise RuntimeError(f"registry config-blob request failed with HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("registry config-blob request failed before an authoritative response") from exc
    if "sha256:" + hashlib.sha256(body).hexdigest() != digest:
        raise RuntimeError("registry config blob does not match its descriptor digest")
    return body


def _validate_descriptor(
    value: Any,
    *,
    label: str,
    media_types: set[str] | None = None,
    maximum_size: int | None = None,
) -> tuple[str, int]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    media_type = value.get("mediaType")
    digest = value.get("digest")
    size = value.get("size")
    if not isinstance(media_type, str) or not media_type:
        raise RuntimeError(f"{label} has an invalid mediaType")
    if media_types is not None and media_type not in media_types:
        raise RuntimeError(f"{label} has an unsupported mediaType")
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise RuntimeError(f"{label} has an invalid digest")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RuntimeError(f"{label} has an invalid size")
    if maximum_size is not None and size > maximum_size:
        raise RuntimeError(f"{label} exceeds its size limit")
    return digest, size


def _validate_image_manifest(manifest: Mapping[str, Any]) -> tuple[str, int]:
    if manifest.get("schemaVersion") != 2:
        raise RuntimeError("OCI image manifest schemaVersion must be 2")
    if manifest.get("mediaType") not in _MANIFEST_MEDIA_TYPES:
        raise RuntimeError("OCI image manifest has an unsupported mediaType")
    config_digest, config_size = _validate_descriptor(
        manifest.get("config"),
        label="OCI image config descriptor",
        media_types=_CONFIG_MEDIA_TYPES,
        maximum_size=_MAX_CONFIG_BYTES,
    )
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise RuntimeError("OCI image manifest has no layers list")
    for layer in layers:
        _validate_descriptor(layer, label="OCI image layer descriptor")
    return config_digest, config_size


def _validate_linux_amd64_manifest(
    registry: str,
    repository: str,
    manifest: Mapping[str, Any],
) -> None:
    digest, size = _validate_image_manifest(manifest)
    body = _get_blob(registry, repository, digest)
    if len(body) != size:
        raise RuntimeError("OCI image config blob does not match its descriptor size")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OCI image config blob is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("OCI image config blob must be an object")
    if value.get("os") != "linux" or value.get("architecture") != "amd64":
        raise RuntimeError("OCI image config is not linux/amd64")
    if value.get("variant") not in (None, "", "v1"):
        raise RuntimeError("OCI image config has an unsupported amd64 variant")


def _manifest_object(body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("registry manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("registry manifest must be a JSON object")
    return value


def _linux_amd64_child(index: Mapping[str, Any]) -> tuple[str, int]:
    if index.get("schemaVersion") != 2:
        raise RuntimeError("OCI image index schemaVersion must be 2")
    if index.get("mediaType") not in _INDEX_MEDIA_TYPES:
        raise RuntimeError("OCI image index has an unsupported mediaType")
    manifests = index.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise RuntimeError("registry index has no manifests list")
    candidates: list[tuple[str, int]] = []
    for manifest in manifests:
        digest, size = _validate_descriptor(
            manifest,
            label="OCI image-index manifest descriptor",
            media_types=_MANIFEST_MEDIA_TYPES,
            maximum_size=_MAX_MANIFEST_BYTES,
        )
        platform = manifest.get("platform")
        if not isinstance(platform, dict):
            raise RuntimeError("OCI image-index descriptor has no platform")
        if not isinstance(platform.get("os"), str) or not isinstance(
            platform.get("architecture"), str
        ):
            raise RuntimeError("OCI image-index descriptor has an invalid platform")
        variant = platform.get("variant")
        if variant is not None and not isinstance(variant, str):
            raise RuntimeError("OCI image-index descriptor has an invalid platform variant")
        if platform.get("os") != "linux" or platform.get("architecture") != "amd64":
            continue
        if platform.get("variant") not in (None, "", "v1"):
            continue
        candidates.append((digest, size))
    if len(set(candidates)) != 1:
        raise RuntimeError("registry index does not have exactly one linux/amd64 child")
    return candidates[0]


def resolve_image_reference(source_image: str) -> dict[str, Any]:
    """Resolve one tag to a content-verified linux/amd64 child manifest."""
    registry, display_registry, repository, reference = parse_image_reference(source_image)
    body, top_digest = get_manifest(registry, repository, reference)
    if reference.startswith("sha256:") and reference != top_digest:
        raise RuntimeError("registry response does not match the requested immutable digest")
    top = _manifest_object(body)
    media_type = top.get("mediaType")
    if media_type in _INDEX_MEDIA_TYPES:
        child_digest, child_size = _linux_amd64_child(top)
        child_body, verified_child_digest = get_manifest(registry, repository, child_digest)
        child = _manifest_object(child_body)
        if verified_child_digest != child_digest:
            raise RuntimeError("registry child digest changed during resolution")
        if len(child_body) != child_size:
            raise RuntimeError("registry child manifest does not match its descriptor size")
        _validate_linux_amd64_manifest(registry, repository, child)
        index_digest: str | None = top_digest
    elif media_type in _MANIFEST_MEDIA_TYPES:
        child_digest = top_digest
        index_digest = None
        _validate_linux_amd64_manifest(registry, repository, top)
    else:
        raise RuntimeError("registry returned an unsupported OCI manifest mediaType")
    return {
        "schema_version": LOCK_SCHEMA,
        "status": "available",
        "source_image_requested": source_image,
        "source_image_resolved": f"{display_registry}/{repository}@{child_digest}",
        "registry": registry,
        "repository": repository,
        "reference": reference,
        "reference_kind": "digest" if reference.startswith("sha256:") else "tag",
        "index_digest": index_digest,
        "child_manifest_digest": child_digest,
        "platform": {"os": "linux", "architecture": "amd64"},
        "resolved_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def _missing_image_lock(source_image: str) -> dict[str, Any]:
    registry, _, repository, reference = parse_image_reference(source_image)
    return {
        "schema_version": LOCK_SCHEMA,
        "status": "missing",
        "source_image_requested": source_image,
        "registry": registry,
        "repository": repository,
        "reference": reference,
        "reference_kind": "digest" if reference.startswith("sha256:") else "tag",
        "platform": {"os": "linux", "architecture": "amd64"},
        "resolved_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def validate_image_lock(value: Mapping[str, Any]) -> None:
    """Validate registry provenance and canonical resolved-reference binding."""
    if value.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("unsupported OCI image-lock schema")
    source_image = _required_text(value, "source_image_requested")
    registry, display_registry, repository, reference = parse_image_reference(source_image)
    expected = {
        "registry": registry,
        "repository": repository,
        "reference": reference,
        "reference_kind": "digest" if reference.startswith("sha256:") else "tag",
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError(f"OCI image lock provenance mismatch for {source_image}")
    resolved_at = value.get("resolved_at")
    if not isinstance(resolved_at, str) or not resolved_at:
        raise ValueError(f"OCI image lock has no resolution timestamp for {source_image}")
    try:
        timestamp = datetime.datetime.fromisoformat(resolved_at)
    except ValueError as exc:
        raise ValueError(f"OCI image lock has an invalid timestamp for {source_image}") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"OCI image lock timestamp has no timezone for {source_image}")
    status = value.get("status")
    if status == "missing":
        if any(
            key in value
            for key in ("source_image_resolved", "index_digest", "child_manifest_digest")
        ):
            raise ValueError(f"missing OCI image lock contains a digest for {source_image}")
        return
    if status != "available":
        raise ValueError(f"OCI image lock has an invalid status for {source_image}")
    child_digest = _required_text(value, "child_manifest_digest")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", child_digest) is None:
        raise ValueError(f"OCI image lock child digest is invalid for {source_image}")
    index_digest = value.get("index_digest")
    if index_digest is not None and (
        not isinstance(index_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", index_digest) is None
    ):
        raise ValueError(f"OCI image lock index digest is invalid for {source_image}")
    if reference.startswith("sha256:") and (
        (index_digest is None and child_digest != reference)
        or (index_digest is not None and index_digest != reference)
    ):
        raise ValueError(f"OCI image lock does not bind the requested digest for {source_image}")
    expected_image = f"{display_registry}/{repository}@{child_digest}"
    if value.get("source_image_resolved") != expected_image:
        raise ValueError(f"OCI image lock resolved reference mismatch for {source_image}")


def _load_locks(path: Path) -> dict[str, dict[str, Any]]:
    locks: dict[str, dict[str, Any]] = {}
    sources = [path] if path.exists() else []
    checkpoint_dir = _checkpoint_directory(path)
    sources.extend(sorted(checkpoint_dir.glob("*.json")))
    for source in sources:
        _validate_private(source)
        for value in _read_jsonl(source):
            validate_image_lock(value)
            source_image = _required_text(value, "source_image_requested")
            previous = locks.get(source_image)
            if previous is None or previous == value:
                locks[source_image] = value
            elif previous.get("status") == "missing" and _lock_is_newer(value, previous):
                locks[source_image] = value
            else:
                raise ValueError(f"conflicting OCI image locks for {source_image}")
    return locks


def _lock_is_newer(candidate: Mapping[str, Any], previous: Mapping[str, Any]) -> bool:
    if candidate.get("source_image_requested") != previous.get("source_image_requested"):
        return False
    candidate_time = datetime.datetime.fromisoformat(_required_text(candidate, "resolved_at"))
    previous_time = datetime.datetime.fromisoformat(_required_text(previous, "resolved_at"))
    return candidate_time > previous_time


def _checkpoint_directory(path: Path) -> Path:
    directory = path.with_name(path.name + ".d")
    _ensure_private_directory(directory)
    return directory


def _write_lock_checkpoint(
    directory: Path,
    lock: Mapping[str, Any],
    *,
    refresh_missing: bool,
) -> None:
    source_image = _required_text(lock, "source_image_requested")
    filename = hashlib.sha256(source_image.encode("utf-8")).hexdigest() + ".json"
    target = directory / filename
    if target.exists():
        _validate_private(target)
        existing = list(_read_jsonl(target))
        if len(existing) == 1:
            validate_image_lock(existing[0])
        if existing == [dict(lock)]:
            return
        if (
            not refresh_missing
            or len(existing) != 1
            or existing[0].get("status") != "missing"
            or not _lock_is_newer(lock, existing[0])
        ):
            raise ValueError(f"conflicting OCI lock checkpoint for {source_image}")
    _atomic_write_jsonl(target, [lock])


def _lock_task(
    row: Mapping[str, Any],
    lock: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _validate_task(row)
    requested = _source_image(row)
    if lock is None:
        raise ValueError(f"SWE task image has no OCI lock: {requested}")
    validate_image_lock(lock)
    if lock.get("status") == "missing":
        raise ImageNotFoundError(f"SWE source image is authoritatively missing: {requested}")
    resolved = _required_text(lock, "source_image_resolved")
    locked = json.loads(json.dumps(row))
    sandbox = dict(_required_mapping(locked, "sandbox"))
    sandbox["source_image"] = resolved
    sandbox["image_lock"] = {
        "schema_version": LOCK_SCHEMA,
        "source_image_requested": requested,
        "source_image_resolved": resolved,
        "input_content_digest": _required_digest(row, "content_digest"),
        "index_digest": lock.get("index_digest"),
        "child_manifest_digest": _required_text(lock, "child_manifest_digest"),
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    locked["sandbox"] = sandbox
    locked.pop("content_digest", None)
    locked["content_digest"] = _stable_digest_without_bindings(locked)
    return locked


def _selected_rows(config: ImageLockConfig) -> Iterable[dict[str, Any]]:
    selected = 0
    for row in _read_jsonl(config.private_manifest):
        if config.instance_id is not None and row.get("instance_id") != config.instance_id:
            continue
        if config.limit is not None and selected >= config.limit:
            break
        selected += 1
        yield row


def _resolve_or_missing(source_image: str) -> dict[str, Any]:
    try:
        return resolve_image_reference(source_image)
    except ImageNotFoundError:
        return _missing_image_lock(source_image)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _resolve_pending_locks(
    pending: list[str],
    *,
    locks: dict[str, dict[str, Any]],
    lock_path: Path,
    concurrency: int,
    batch_size: int,
    refresh_missing: bool,
) -> int:
    resolved = 0
    checkpoint_dir = _checkpoint_directory(lock_path)
    for batch in _chunks(pending, batch_size):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(_resolve_or_missing, batch))
        for source_image, lock in zip(batch, results, strict=True):
            validate_image_lock(lock)
            if _required_text(lock, "source_image_requested") != source_image:
                raise RuntimeError("OCI resolver returned a lock for a different source image")
        for lock in results:
            _write_lock_checkpoint(
                checkpoint_dir,
                lock,
                refresh_missing=refresh_missing,
            )
            locks[_required_text(lock, "source_image_requested")] = lock
            resolved += 1
    return resolved


def lock_private_tasks(config: ImageLockConfig) -> dict[str, int]:
    """Resolve image refs, quarantine 404s, and stream an immutable task manifest."""
    checkpoint_dir = config.image_lock_manifest.with_name(config.image_lock_manifest.name + ".d")
    _require_distinct_paths(
        config.private_manifest,
        config.locked_manifest,
        config.image_lock_manifest,
        checkpoint_dir,
    )
    _validate_private(config.private_manifest)
    input_fingerprint = _capture_private_fingerprint(config.private_manifest)
    if config.concurrency <= 0 or config.checkpoint_batch_size <= 0:
        raise ValueError("OCI lock concurrency and checkpoint batch size must be positive")
    if config.refresh_missing and not config.resolve_missing:
        raise ValueError("refreshing missing OCI locks requires resolution to be enabled")
    locks = _load_locks(config.image_lock_manifest)
    references: set[str] = set()
    selected = 0
    for row in _selected_rows(config):
        _validate_task(row)
        references.add(_source_image(row))
        selected += 1
    if selected == 0:
        raise ValueError("private SWE manifest has no selected tasks")
    unresolved = references - locks.keys()
    cached_missing = {
        source_image
        for source_image in references
        if locks.get(source_image, {}).get("status") == "missing"
    }
    pending = sorted(unresolved | (cached_missing if config.refresh_missing else set()))
    if pending and not config.resolve_missing:
        raise ValueError(f"{len(pending)} SWE task image references have no OCI lock")
    resolved = _resolve_pending_locks(
        pending,
        locks=locks,
        lock_path=config.image_lock_manifest,
        concurrency=config.concurrency,
        batch_size=config.checkpoint_batch_size,
        refresh_missing=config.refresh_missing,
    )
    def unchanged() -> None:
        _assert_private_unchanged(config.private_manifest, input_fingerprint)
    _atomic_write_jsonl(
        config.image_lock_manifest,
        locks.values(),
        before_replace=unchanged,
    )
    emitted = 0
    missing = 0

    def locked_rows() -> Iterable[dict[str, Any]]:
        nonlocal emitted, missing
        for row in _selected_rows(config):
            lock = locks[_source_image(row)]
            if lock.get("status") == "missing":
                missing += 1
                continue
            emitted += 1
            yield _lock_task(row, lock)

    _atomic_write_jsonl(
        config.locked_manifest,
        locked_rows(),
        before_replace=unchanged,
    )
    return {
        "tasks": selected,
        "emitted": emitted,
        "missing": missing,
        "resolved": resolved,
        "reused": len(references) - resolved,
        "refreshed_missing": len(cached_missing) if config.refresh_missing else 0,
        "cached_missing": 0 if config.refresh_missing else len(cached_missing),
    }


def _validate_task(row: Mapping[str, Any]) -> None:
    if row.get("schema_version") != TASK_SCHEMA:
        raise ValueError("unsupported private SWE task schema")
    task_digest = _required_digest(row, "task_digest")
    content_digest = _required_digest(row, "content_digest")
    if not task_digest or _stable_digest_without_bindings(row) != content_digest:
        raise ValueError("private SWE task digest binding is invalid")
    parse_image_reference(_source_image(row))
    validate_task_image_policy(row)


def _source_image(row: Mapping[str, Any]) -> str:
    return _required_text(_required_mapping(row, "sandbox"), "source_image")


def _stable_digest_without_bindings(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_digest", None)
    payload.pop("task_digest", None)
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be an object")
    return result


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be non-empty text")
    return result


def _required_digest(value: Mapping[str, Any], key: str) -> str:
    result = _required_text(value, key)
    if _DIGEST.fullmatch(result) is None:
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return result


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    _validate_private(path)
    before = os.lstat(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or opened.st_mode & 0o077
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            != (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        ):
            raise RuntimeError(f"private JSONL changed before open: {path}")
        handle = os.fdopen(descriptor, "r", encoding="utf-8", closefd=False)
        try:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"private JSONL row {line_number} must be an object")
                yield value
        finally:
            handle.close()
            after = os.fstat(descriptor)
            try:
                path_after = os.lstat(path)
            except FileNotFoundError as exc:
                raise RuntimeError(f"private JSONL disappeared during read: {path}") from exc
            if (
                (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
            ):
                raise RuntimeError(f"private JSONL changed during read: {path}")
    finally:
        os.close(descriptor)


def _atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    before_replace: Callable[[], None] | None = None,
) -> None:
    _ensure_private_directory(path.parent)
    if path.exists() or path.is_symlink():
        metadata = os.lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise PermissionError(f"private output target must be a regular non-symlink: {path}")
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise PermissionError(f"private output target is not owner-only: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_private(path: Path) -> None:
    metadata = os.lstat(path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise PermissionError(f"private input must be a regular non-symlink: {path}")
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise PermissionError(f"private file must be owner-only and owned by this user: {path}")


def _capture_private_fingerprint(path: Path) -> _PrivateFileFingerprint:
    _validate_private(path)
    before = os.lstat(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or opened.st_mode & 0o077
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            != (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        ):
            raise RuntimeError(f"private input changed before fingerprinting: {path}")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        try:
            path_after = os.lstat(path)
        except FileNotFoundError as exc:
            raise RuntimeError(f"private input disappeared during fingerprinting: {path}") from exc
        if (
            size != after.st_size
            or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
            != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino)
        ):
            raise RuntimeError(f"private input changed during fingerprinting: {path}")
    finally:
        os.close(descriptor)
    return _PrivateFileFingerprint(
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
        sha256=digest.hexdigest(),
    )


def _assert_private_unchanged(path: Path, expected: _PrivateFileFingerprint) -> None:
    actual = _capture_private_fingerprint(path)
    if actual != expected:
        raise RuntimeError(f"private input changed during OCI image locking: {path}")


def _require_distinct_paths(*paths: Path) -> None:
    normalized: dict[Path, Path] = {}
    inodes: dict[tuple[int, int], Path] = {}
    for path in paths:
        resolved = path.resolve(strict=False)
        previous = normalized.get(resolved)
        if previous is not None:
            raise ValueError(f"private OCI input/output paths alias: {previous} and {path}")
        normalized[resolved] = path
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        inode = (metadata.st_dev, metadata.st_ino)
        previous = inodes.get(inode)
        if previous is not None:
            raise ValueError(f"private OCI input/output paths are hard-linked: {previous} and {path}")
        inodes[inode] = path


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError(f"private output parent must be a non-symlink directory: {path}")
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"private output parent is owned by another user: {path}")
    os.chmod(path, 0o700)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--image-lock-manifest", type=Path, required=True)
    parser.add_argument("--resolve-missing", action="store_true")
    parser.add_argument(
        "--refresh-missing",
        action="store_true",
        help="retry only cached 404 locks; cached missing evidence is otherwise stable",
    )
    parser.add_argument("--instance-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--checkpoint-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = _parse_args()
    summary = lock_private_tasks(
        ImageLockConfig(
            private_manifest=args.private_manifest,
            locked_manifest=args.locked_manifest,
            image_lock_manifest=args.image_lock_manifest,
            resolve_missing=args.resolve_missing,
            refresh_missing=args.refresh_missing,
            instance_id=args.instance_id,
            limit=args.limit,
            concurrency=args.concurrency,
            checkpoint_batch_size=args.checkpoint_batch_size,
        )
    )
    print(
        "SWE OCI image lock complete: "
        f"tasks={summary['tasks']} emitted={summary['emitted']} missing={summary['missing']} "
        f"resolved={summary['resolved']} reused={summary['reused']}"
        f" cached_missing={summary['cached_missing']} "
        f"refreshed_missing={summary['refreshed_missing']}"
    )


if __name__ == "__main__":
    main()
