from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from ml_product_management.config import PipelineError, S3Settings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


class S3Storage:
    def __init__(self, client: "S3Client", bucket: str) -> None:
        self._client = client
        self.bucket = bucket

    @classmethod
    def from_settings(cls, settings: S3Settings, max_connections: int = 10) -> "S3Storage":
        client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key,
            region_name=settings.region,
            config=Config(
                max_pool_connections=max_connections,
                s3={"addressing_style": "path"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )
        return cls(client, settings.bucket)

    @classmethod
    def from_env(cls, max_connections: int = 10) -> "S3Storage":
        return cls.from_settings(S3Settings.from_env(), max_connections)

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] in NOT_FOUND_CODES:
                return False
            raise
        return True

    def prefix_exists(self, prefix: str) -> bool:
        response = self._client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
        return bool(response.get("Contents"))

    def upload_file(self, path: Path, key: str) -> None:
        self._client.upload_file(str(path), self.bucket, key)

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def download_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        try:
            self._client.download_file(self.bucket, key, str(partial))
        except ClientError as error:
            partial.unlink(missing_ok=True)
            raise PipelineError(f"Не удалось скачать {self.uri(key)}: {error}") from error
        partial.replace(destination)
