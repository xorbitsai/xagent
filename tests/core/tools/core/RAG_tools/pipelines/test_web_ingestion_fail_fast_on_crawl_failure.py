import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import WebCrawlConfig
from xagent.core.tools.core.RAG_tools.pipelines import web_ingestion


@pytest.mark.asyncio
async def test_run_web_ingestion_fail_fast_when_no_pages_crawled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyCrawler:
        def __init__(self, *_args, **_kwargs) -> None:
            self.failed_urls = {"https://example.com": "Insufficient content"}
            self.total_urls_found = 1

        async def crawl(self):
            return []

    monkeypatch.setattr(web_ingestion, "WebCrawler", DummyCrawler)

    res = await web_ingestion.run_web_ingestion(
        collection="c",
        crawl_config=WebCrawlConfig(
            start_url="https://example.com", max_pages=1, max_depth=1
        ),
        ingestion_config=None,
        user_id=1,
        is_admin=False,
        trace_id="t",
    )

    assert res.status == "error"
    assert res.pages_crawled == 0
    assert res.pages_failed == 1
    assert res.documents_created == 0
    assert "Website crawling failed" in res.message
    assert "Insufficient content" in res.message
