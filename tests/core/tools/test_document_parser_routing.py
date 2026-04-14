import pytest

from xagent.core.tools.core.document_parser import (
    DocumentCapabilities,
    DocumentParseArgs,
    parse_document,
)
from xagent.providers.pdf_parser.deepdoc import DeepDocParser


@pytest.mark.asyncio
async def test_parse_document_prefers_deepdoc_for_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "table.csv"
    csv_path.write_text("a,b\n1,2\n")

    captured: dict[str, str] = {}

    async def _fake_parse(self, file_path: str, **kwargs):
        captured["file_path"] = file_path
        return {"text_segments": [], "figures": [], "tables": []}

    monkeypatch.setattr(DeepDocParser, "parse", _fake_parse)

    await parse_document(
        DocumentParseArgs(
            file_path=str(csv_path),
            parser_name=None,
            capabilities=DocumentCapabilities(),
        )
    )

    assert captured["file_path"] == str(csv_path)


@pytest.mark.asyncio
async def test_parse_document_prefers_deepdoc_for_xlsx(monkeypatch, tmp_path):
    xlsx_path = tmp_path / "table.xlsx"
    xlsx_path.write_text("fake xlsx content")

    captured: dict[str, str] = {}

    async def _fake_parse(self, file_path: str, **kwargs):
        captured["file_path"] = file_path
        return {"text_segments": [], "figures": [], "tables": []}

    monkeypatch.setattr(DeepDocParser, "parse", _fake_parse)

    await parse_document(
        DocumentParseArgs(
            file_path=str(xlsx_path),
            parser_name=None,
            capabilities=DocumentCapabilities(),
        )
    )

    assert captured["file_path"] == str(xlsx_path)
