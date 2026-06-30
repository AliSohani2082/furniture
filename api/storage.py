# api/storage.py
from __future__ import annotations

import io
import os

import boto3
from botocore.client import Config

MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_BUCKET = os.environ["MINIO_BUCKET"]
MINIO_ROOT_USER = os.environ["MINIO_ROOT_USER"]
MINIO_ROOT_PASSWORD = os.environ["MINIO_ROOT_PASSWORD"]

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
    return _client


def ensure_bucket() -> None:
    client = _get_client()
    try:
        client.head_bucket(Bucket=MINIO_BUCKET)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=MINIO_BUCKET)


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    _get_client().upload_fileobj(
        io.BytesIO(data),
        MINIO_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return key


def presigned_url(key: str, expiry: int = 86400) -> str:
    return _get_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": MINIO_BUCKET, "Key": key},
        ExpiresIn=expiry,
    )
