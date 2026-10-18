"""Where a capture lives: a local directory or an S3 bucket, behind pyarrow's filesystem layer.

    location("runs/run1")                                  -> local files runs/run1.*
    location("s3://bucket/prefix/run1")                    -> objects prefix/run1.* in bucket
    location("run1", bucket="bucket", prefix="prefix")     -> the same
    s3={"region": ..., "endpoint_override": ..., "access_key": ..., ...}   -> pyarrow.fs.S3FileSystem options
      (credentials otherwise come from the usual AWS environment / profile / instance role)

S3 fits the layout: tensor files are immutable once closed (each flush closes the current one, so a capture
is readable mid-run), the Parquet tables are small whole-object rewrites, and blob reads are offset-plus-length
range requests.
"""
import os
from dataclasses import dataclass

import pyarrow.fs as pafs

S3_ROTATE_BYTES = 1 << 30          # objects of at most 1 GiB: a few multipart parts, cheap to re-upload or re-fetch
LOCAL_ROTATE_BYTES = 4 << 30


@dataclass
class Location:
    fs: pafs.FileSystem
    path: str            # the stem's path inside fs (no scheme)
    remote: bool
    uri: str             # how to print it

    @property
    def rotate_default(self):
        return S3_ROTATE_BYTES if self.remote else LOCAL_ROTATE_BYTES

    def exists(self, path):
        return self.fs.get_file_info(path).type != pafs.FileType.NotFound

    def ensure_dir(self):
        if not self.remote:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)


def location(stem, bucket=None, prefix=None, s3=None):
    stem = str(stem)
    if bucket is None and stem.startswith("s3://"):
        bucket, _, rest = stem[5:].partition("/")
        prefix, _, stem = rest.rpartition("/")
    if bucket is None:
        path = os.path.abspath(stem)
        return Location(pafs.LocalFileSystem(), path, False, path)
    if not stem or "/" in stem.strip("/"):
        raise ValueError(f"the stem on S3 must be a plain name (got {stem!r}); put directories in prefix=")
    opts = dict(s3 or {})
    if "region" not in opts and "endpoint_override" not in opts:
        try:
            opts["region"] = pafs.resolve_s3_region(bucket)
        except Exception:
            pass
    key = "/".join(p for p in [(prefix or "").strip("/"), stem.strip("/")] if p)
    return Location(pafs.S3FileSystem(**opts), f"{bucket}/{key}", True, f"s3://{bucket}/{key}")
