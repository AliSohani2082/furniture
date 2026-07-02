# api/tests/test_storage.py
# Requires MinIO running at localhost:9000 with minioadmin/changeme
# Run: docker compose up storage -d

import os
os.environ["MINIO_ENDPOINT"] = "http://localhost:9000"
os.environ["MINIO_BUCKET"] = "test-furnitur-storage"
os.environ["MINIO_ROOT_USER"] = "minioadmin"
os.environ["MINIO_ROOT_PASSWORD"] = "changeme"

import httpx
import pytest


def test_upload_and_presign():
    from storage import ensure_bucket, upload_bytes, presigned_url

    ensure_bucket()
    key = upload_bytes("tests/hello.txt", b"hello world", "text/plain")
    assert key == "tests/hello.txt"

    url = presigned_url("tests/hello.txt")
    resp = httpx.get(url)
    assert resp.status_code == 200
    assert resp.content == b"hello world"


def test_upload_binary():
    from storage import ensure_bucket, upload_bytes, presigned_url

    ensure_bucket()
    data = bytes(range(256))
    upload_bytes("tests/binary.bin", data, "application/octet-stream")
    url = presigned_url("tests/binary.bin")
    resp = httpx.get(url)
    assert resp.content == data
