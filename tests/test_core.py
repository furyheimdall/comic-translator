import io
import zipfile

import pytest
from PIL import Image

from app.jobs import IngestError, collect_images
from app.main import _inject_system
from app.providers import strip_code_fence, text_only, to_anthropic, to_responses_input

PNG = "data:image/png;base64,iVBORw0KGgo="


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, "PNG")
    return buffer.getvalue()


def zip_bytes(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, png_bytes() if name.endswith(".png") else b"x")
    return buffer.getvalue()


def test_archive_pages_are_natural_sorted_and_junk_skipped():
    data = zip_bytes(["vol/p10.png", "vol/p2.png", "__MACOSX/vol/._p2.png", "vol/.hidden.png", "vol/notes.txt", "vol/p1.png"])
    names = [name for name, _ in collect_images([("vol.cbz", io.BytesIO(data))])]
    assert names == ["vol/p1.png", "vol/p2.png", "vol/p10.png"]


def test_uploads_are_ordered_by_natural_file_name():
    uploads = [(name, io.BytesIO(png_bytes())) for name in ("page10.png", "page9.png", "page1.png")]
    assert [name for name, _ in collect_images(uploads)] == ["page1.png", "page9.png", "page10.png"]


def pdf_from_images(images: list[Image.Image], resolution: float) -> bytes:
    buffer = io.BytesIO()
    images[0].save(buffer, "PDF", save_all=True, append_images=images[1:], resolution=resolution)
    return buffer.getvalue()


def test_scanned_pdf_pages_keep_native_resolution_and_order():
    pages = [Image.new("RGB", (1200, 1700), color) for color in ("white", "black")]
    # 150 DPI page box is smaller than the pixel size; extraction must not resample.
    result = collect_images([("vol 1.pdf", io.BytesIO(pdf_from_images(pages, 150)))])
    assert [name for name, _ in result] == ["vol 1/p0001.png", "vol 1/p0002.png"]
    decoded = [Image.open(io.BytesIO(data)) for _, data in result]
    assert [img.size for img in decoded] == [(1200, 1700), (1200, 1700)]
    assert [img.getpixel((10, 10)) for img in decoded] == [(255, 255, 255), (0, 0, 0)]


def test_pdf_page_without_full_page_raster_is_rendered_within_bounds():
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument.new()
    page = document.new_page(2000, 3000)  # points; 300 DPI would exceed the cap
    stamp = pdfium.PdfImage.new(document)
    stamp.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", (100, 100), "red")))
    stamp.set_matrix(pdfium.PdfMatrix().scale(100, 100).translate(50, 50))
    page.insert_obj(stamp)
    page.gen_content()
    buffer = io.BytesIO()
    document.save(buffer)
    [(_, data)] = collect_images([("doc.pdf", io.BytesIO(buffer.getvalue()))])
    assert max(Image.open(io.BytesIO(data)).size) == 3000


def test_broken_pdf_is_rejected():
    with pytest.raises(IngestError, match="PDF를 열 수 없습니다"):
        collect_images([("bad.pdf", io.BytesIO(b"%PDF-1.7 garbage"))])


@pytest.mark.parametrize(
    "uploads, message",
    [
        ([("a.gif", io.BytesIO(b"GIF89a"))], "지원하지 않는"),
        ([("a.png", io.BytesIO(b"not an image"))], "읽을 수 없습니다"),
        ([("a.zip", io.BytesIO(b"PK broken"))], "손상된"),
        ([("a.zip", io.BytesIO(zip_bytes(["readme.txt"])))], "이미지가 없습니다"),
    ],
)
def test_bad_uploads_are_rejected(uploads, message):
    with pytest.raises(IngestError, match=message):
        collect_images(uploads)


def test_anthropic_conversion_moves_system_and_images():
    system, messages = to_anthropic(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": PNG}}]},
        ]
    )
    assert system == "sys"
    assert messages[0]["content"][1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}


def test_responses_conversion_uses_input_and_output_parts():
    instructions, items = to_responses_input(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert instructions == "sys"
    assert items[0]["content"] == [{"type": "input_text", "text": "q"}]
    assert items[1]["content"] == [{"type": "output_text", "text": "a"}]


def test_text_only_flattens_multimodal_content():
    [message] = text_only([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": PNG}}, {"type": "text", "text": "1: x"}]}])
    assert message["content"] == "1: x"


@pytest.mark.parametrize(
    "raw, expected",
    [('```json\n{"a": 1}\n```', '{"a": 1}'), ('{"a": 1}', '{"a": 1}'), ("```\n[1]\n```\n", "[1]")],
)
def test_strip_code_fence(raw, expected):
    assert strip_code_fence(raw) == expected


def test_inject_system_appends_to_existing_or_prepends():
    assert _inject_system([{"role": "system", "content": "a"}, {"role": "user", "content": "u"}], "g")[0]["content"] == "a\n\ng"
    assert _inject_system([{"role": "user", "content": "u"}], "g")[0] == {"role": "system", "content": "g"}


def test_login_accepts_utf8_password_and_rejects_mismatch(tmp_path):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.settings import load_settings

    credential = '시험용 "비밀번호" \\ $literal % `ticks`'
    settings = replace(load_settings(), data_dir=tmp_path, host="127.0.0.1", password=credential)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/jobs").status_code == 401
        assert client.post("/api/login", json={"password": credential + "틀림"}).status_code == 401
        assert client.get("/api/jobs").status_code == 401
        assert client.post("/api/login", json={"password": credential}).status_code == 200
        assert client.get("/api/jobs").json() == []
