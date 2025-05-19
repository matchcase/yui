from typing import Dict, Any, Optional
import html2text
from playwright.async_api import async_playwright
import logging
from tools import register_tool
from langchain_community.utilities import SearxSearchWrapper
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# SearXNG instance URL
SEARXNG_URL = "http://localhost:8888"

class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query.")
    limit: Optional[int] = Field(description="The maximum number of results.",
                                 default=5)

@register_tool(
    name="web_search",
    description="Search the web SearxNG.",
    args_schema=WebSearchArgs
)

async def web_search(query: str, limit: int = 5) -> Dict[str, Any]:
    search = SearxSearchWrapper(searx_host="http://localhost:8888")
    results = search.results(query, num_results=limit)
    return results

class CrawlPageArgs(BaseModel):
    url: str = Field(description="The URL to be crawled.")

@register_tool(
    name="crawl_page",
    description="Crawl a webpage for content.",
    args_schema=CrawlPageArgs
)
async def crawl_page(url: str) -> Dict[str, Any]:
    logger.info(f"Crawling URL: {url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, timeout=60000)
            content = await page.content()
            text_content = html2text.html2text(content)
            await browser.close()
            return {
                "url": url,
                "content_snippet": text_content[:2000] # Trim to not clog the context window
            }
            logger.info(f"Successfully crawled {url}")
    except Exception as e:
        logger.error(f"Error crawling {url}: {e}")
        return {"error": f"Failed to crawl {url}: {str(e)}"}

