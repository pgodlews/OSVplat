#!/usr/bin/env python3
"""Presigned S3 URLs for a rented-GPU run: one to fetch the clip, one to put
the result. Run on your own machine; the URLs are the only thing that goes to
the rented host, and each can reach one object until it expires.

    pip install boto3
    scripts/presign_s3.py s3://my-bucket/osvplat/run1 --clip ~/clips/DJI_0198.OSV

uploads the clip to .../run1/<clip name> (skip with --no-upload if it is
already there), then prints the environment for the container:

    INPUT_URL=...            presigned GET of the clip
    INPUT_SHA256=...         checked by the container before it uses the clip
    OUTPUT_UPLOAD_URL=...    presigned PUT of .../run1/result.tar

--expires (hours, default 24) must outlast the whole run: pull, queue, upload.
SigV4 allows up to 7 days; temporary credentials (SSO, roles) cap it at their
own expiry: a URL signed with them dies with the session. --endpoint-url for R2, B2, MinIO, versitygw and other S3-compatible
stores (path-style URLs, SigV4). Uses your normal AWS credentials, or
--env-file with ROOT_ACCESS_KEY=/ROOT_SECRET_KEY= (or AWS_ACCESS_KEY_ID=/
AWS_SECRET_ACCESS_KEY=) lines; they stay here and are never printed.

    scripts/presign_s3.py s3://inputs/run1 --clip x.OSV \
        --result-prefix s3://results/run1 \
        --endpoint-url https://s3.example.com --env-file ~/.config/s3.env
"""
import argparse
import hashlib
import sys
from pathlib import Path
from urllib.parse import urlsplit


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prefix", help="s3://bucket/some/prefix for this run")
    ap.add_argument("--clip", type=Path, help="local clip for INPUT_URL")
    ap.add_argument("--no-upload", action="store_true",
                    help="the clip is already at <prefix>/<clip name>")
    ap.add_argument("--result", default="result.tar",
                    help="object name for OUTPUT_UPLOAD_URL (default result.tar)")
    ap.add_argument("--result-prefix",
                    help="s3://bucket/prefix for the result, if not the clip's")
    ap.add_argument("--expires", type=float, default=24, help="hours (default 24)")
    ap.add_argument("--endpoint-url")
    ap.add_argument("--region")
    ap.add_argument("--env-file", type=Path, help="credentials file (see above)")
    a = ap.parse_args()

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        sys.exit("needs boto3: pip install boto3 (and botocore[crt] for `aws login` credentials)")

    def split(s3url):
        u = urlsplit(s3url)
        if u.scheme != "s3" or not u.netloc:
            sys.exit(f"{s3url}: must look like s3://bucket/path")
        return u.netloc, u.path.strip("/")
    bucket, prefix = split(a.prefix)
    rbucket, rprefix = split(a.result_prefix) if a.result_prefix else (bucket, prefix)
    key = lambda name: f"{prefix}/{name}" if prefix else name     # noqa: E731
    rkey = f"{rprefix}/{a.result}" if rprefix else a.result
    seconds = int(a.expires * 3600)
    if not 60 <= seconds <= 7 * 24 * 3600:
        sys.exit("--expires must be between 1 minute and 7 days")

    creds = {}
    if a.env_file:
        env = dict(l.split("=", 1) for l in a.env_file.expanduser().read_text().split() if "=" in l)
        creds = {"aws_access_key_id": env.get("ROOT_ACCESS_KEY") or env.get("AWS_ACCESS_KEY_ID"),
                 "aws_secret_access_key": env.get("ROOT_SECRET_KEY") or env.get("AWS_SECRET_ACCESS_KEY")}
        if not all(creds.values()):
            sys.exit(f"{a.env_file}: no access key / secret key lines")
    # Path-style for S3-compatible servers: bucket.host names need wildcard DNS.
    cfg = Config(signature_version="s3v4",
                 s3={"addressing_style": "path"} if a.endpoint_url else {})
    s3 = boto3.client("s3", endpoint_url=a.endpoint_url,
                      region_name=a.region or ("us-east-1" if a.endpoint_url else None),
                      config=cfg, **creds)
    lines = []
    if a.clip:
        if not a.clip.is_file():
            sys.exit(f"no such clip: {a.clip}")
        k = key(a.clip.name)
        if not a.no_upload:
            print(f"uploading {a.clip} to s3://{bucket}/{k}", file=sys.stderr)
            s3.upload_file(str(a.clip), bucket, k)
        url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": k},
                                        ExpiresIn=seconds)
        lines += [f"INPUT_URL={url}", f"INPUT_SHA256={sha256(a.clip)}"]
    # No ContentType/ContentMD5 in Params: signed headers would have to match
    # exactly. The container sends no Content-MD5 (AWS refuses unsigned extra
    # headers on a presigned PUT); it checks the returned ETag instead.
    url = s3.generate_presigned_url("put_object", Params={"Bucket": rbucket, "Key": rkey},
                                    ExpiresIn=seconds)
    lines.append(f"OUTPUT_UPLOAD_URL={url}")
    print("\n".join(lines))
    print(f"result: s3://{rbucket}/{rkey} (check its sha256 against the "
          f"container log or GET /api/jobs/<id>)", file=sys.stderr)


if __name__ == "__main__":
    main()
