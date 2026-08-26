#!/usr/bin/env python3
"""Serve the local RoboMIND robot-only annotation website.

This dependency-light server validates every submitted mask and writes only to
the annotation package's declared ``robot_only_masks`` directory.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import mimetypes
import os
import re
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = PROJECT_ROOT / "outputs" / "actmask" / "rm_geometry_mask_audit" / "manual_reference_plan"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class AnnotationStore:
    """Validated bridge between the immutable manifest and annotation outputs."""

    def __init__(self, package: Path) -> None:
        self.package = package.resolve()
        self.manifest_path = self.package / "annotation_manifest.jsonl"
        self.progress_path = self.package / "annotation_progress.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"annotation manifest not found: {self.manifest_path}")
        self.rows = [json.loads(line) for line in self.manifest_path.read_text().splitlines() if line]
        if len(self.rows) != 200:
            raise RuntimeError(f"expected 200 annotation frames, found {len(self.rows)}")
        self.by_id = {str(row["id"]): row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise RuntimeError("annotation manifest contains duplicate ids")
        self.lock = threading.RLock()
        self.mask_root = self.package / "robot_only_masks"
        self.mask_root.mkdir(parents=True, exist_ok=True)

    def _declared_path(self, row: dict[str, Any], key: str) -> Path:
        path = (self.package / str(row[key])).resolve()
        if self.package not in path.parents:
            raise RuntimeError("manifest path escaped the annotation package")
        if key == "robot_only_mask_path" and path.parent != self.mask_root.resolve():
            raise RuntimeError("manifest mask path escaped robot_only_masks")
        return path

    def row(self, identifier: str) -> dict[str, Any]:
        if not SAFE_ID.fullmatch(identifier) or identifier not in self.by_id:
            raise KeyError(identifier)
        return self.by_id[identifier]

    def image_path(self, identifier: str) -> Path:
        path = self._declared_path(self.row(identifier), "frame_path")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def mask_path(self, identifier: str) -> Path:
        return self._declared_path(self.row(identifier), "robot_only_mask_path")

    def progress(self) -> dict[str, Any]:
        if not self.progress_path.is_file():
            return {"schema": "robomind-annotation-progress-v1", "records": {}}
        value = json.loads(self.progress_path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("records", {}), dict):
            raise RuntimeError("annotation progress schema is invalid")
        return value

    def status_rows(self) -> list[dict[str, Any]]:
        records = self.progress().get("records", {})
        rows = []
        for row in self.rows:
            identifier = str(row["id"])
            result = dict(row)
            record = records.get(identifier, {})
            result.update({
                "completed": self.mask_path(identifier).is_file(),
                "saved_at": record.get("saved_at"),
                "annotator": record.get("annotator", ""),
                "note": record.get("note", ""),
                "foreground_pixels": record.get("foreground_pixels"),
            })
            rows.append(result)
        return rows

    def _write_progress(self, identifier: str, entry: dict[str, Any]) -> None:
        progress = self.progress()
        records = dict(progress.get("records", {}))
        records[identifier] = entry
        value = {"schema": "robomind-annotation-progress-v1", "updated_at": utc_now(), "records": records}
        temporary = self.progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.progress_path)

    def save_mask(self, identifier: str, data_url: str, annotator: str, note: str) -> dict[str, Any]:
        row = self.row(identifier)
        prefix = "data:image/png;base64,"
        if not isinstance(data_url, str) or not data_url.startswith(prefix):
            raise ValueError("标注必须是 PNG data URL")
        try:
            payload = base64.b64decode(data_url[len(prefix):], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("标注数据不是有效的 Base64") from exc
        if len(payload) > MAX_REQUEST_BYTES:
            raise ValueError("标注 PNG 超出大小限制")
        try:
            grayscale = np.asarray(Image.open(io.BytesIO(payload)).convert("L"))
        except Exception as exc:
            raise ValueError("标注数据不是可读取的 PNG") from exc
        expected = (int(row["height"]), int(row["width"]))
        if grayscale.shape != expected:
            raise ValueError(f"标注尺寸 {grayscale.shape} 与期望尺寸 {expected} 不一致")
        binary = np.where(grayscale >= 128, 255, 0).astype(np.uint8)
        foreground = int((binary == 255).sum())
        if foreground < 16:
            raise ValueError("机器人前景为空或过小，请先绘制可见机械臂/夹爪")
        target = self.mask_path(identifier)
        temporary = target.with_suffix(".tmp.png")
        Image.fromarray(binary, mode="L").save(temporary, format="PNG")
        os.replace(temporary, target)
        content = target.read_bytes()
        saved_at = utc_now()
        self._write_progress(identifier, {
            "saved_at": saved_at,
            "annotator": str(annotator).strip()[:80],
            "note": str(note).strip()[:1000],
            "foreground_pixels": foreground,
            "mask_sha256": sha256_bytes(content),
        })
        return {"id": identifier, "foreground_pixels": foreground, "mask_sha256": sha256_bytes(content), "saved_at": saved_at}


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "RoboMINDAnnotation/1.0"

    @property
    def store(self) -> AnnotationStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{utc_now()}] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, cacheable: bool = False) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cacheable else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status)

    def identifier(self, prefix: str) -> str:
        return unquote(urlparse(self.path).path[len(prefix):])

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in {"/", "/index.html"}:
                self.send_file(WEB_ROOT / "robomind_annotation_index.html")
            elif path == "/static/app.css":
                self.send_file(WEB_ROOT / "robomind_annotation_app.css", cacheable=True)
            elif path == "/static/app.js":
                self.send_file(WEB_ROOT / "robomind_annotation_app.js", cacheable=True)
            elif path == "/api/health":
                self.send_json({"status": "ok", "frames": len(self.store.rows), "package": str(self.store.package)})
            elif path == "/api/manifest":
                rows = self.store.status_rows()
                self.send_json({"schema": "robomind-annotation-ui-manifest-v1", "rows": rows, "completed": sum(row["completed"] for row in rows), "total": len(rows)})
            elif path.startswith("/api/image/"):
                self.send_file(self.store.image_path(self.identifier("/api/image/")), cacheable=True)
            elif path.startswith("/api/mask/"):
                mask = self.store.mask_path(self.identifier("/api/mask/"))
                if not mask.is_file():
                    self.error_json(HTTPStatus.NOT_FOUND, "当前帧还没有保存标注")
                else:
                    self.send_file(mask)
            elif path == "/api/progress":
                self.send_json(self.store.progress())
            else:
                self.error_json(HTTPStatus.NOT_FOUND, "未找到请求的页面或接口")
        except KeyError:
            self.error_json(HTTPStatus.NOT_FOUND, "未知的标注帧")
        except FileNotFoundError:
            self.error_json(HTTPStatus.NOT_FOUND, "请求的文件不存在")
        except Exception as exc:
            self.error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.startswith("/api/save/"):
            self.error_json(HTTPStatus.NOT_FOUND, "未找到请求的接口")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("请求体大小无效")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            with self.store.lock:
                result = self.store.save_mask(self.identifier("/api/save/"), value.get("png_base64", ""), value.get("annotator", ""), value.get("note", ""))
            self.send_json({"ok": True, "result": result})
        except KeyError:
            self.error_json(HTTPStatus.NOT_FOUND, "未知的标注帧")
        except (ValueError, json.JSONDecodeError) as exc:
            self.error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self.error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 RoboMIND 中文机器人前景标注网站")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE, help="包含 annotation_manifest.jsonl 的标注包目录")
    parser.add_argument("--host", default="127.0.0.1", help="默认仅本机访问；远程端口映射时显式使用 0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = AnnotationStore(args.package)
    server = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    server.store = store  # type: ignore[attr-defined]
    print(f"标注网站已启动：http://{args.host}:{args.port}")
    print(f"标注包：{store.package}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n标注网站已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
