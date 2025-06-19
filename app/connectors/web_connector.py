import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse
import time

from crawl4ai import AsyncWebCrawler

from connectors.base_connector import BaseConnector
from models.document import Document, DocumentMetadata

logger = logging.getLogger(__name__)

class WebConnector(BaseConnector):
    """Connector for web content using crawl4ai"""
    
    def __init__(self):
        super().__init__()
        self.crawler = None
        self.rate_limit = self.settings.CRAWL_RATE_LIMIT
        self.timeout = self.settings.CRAWL_TIMEOUT
        self.max_pages = self.settings.CRAWL_MAX_PAGES
        self.user_agent = self.settings.CRAWL_USER_AGENT
        self.respect_robots = self.settings.CRAWL_RESPECT_ROBOTS
        self.max_depth = self.settings.CRAWL_MAX_DEPTH
        self._connected = False
        self._last_request_time = 0
    
    async def connect(self) -> bool:
        """Initialize the web crawler"""
        try:
            self.crawler = AsyncWebCrawler(
                verbose=False,
                headless=True,
                browser_type="chromium"
            )
            await self.crawler.start()
            self._connected = True
            logger.info("Connected to web crawler")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to web crawler: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close the web crawler"""
        try:
            if self.crawler:
                await self.crawler.close()
            self._connected = False
            logger.info("Disconnected from web crawler")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting web crawler: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> List[Document]:
        """Fetch web content from URLs"""
        if not self._connected:
            await self.connect()
        
        urls = kwargs.get('urls', [])
        if not urls:
            logger.warning("No URLs provided for web crawling")
            return []
        
        documents = []
        crawled_count = 0
        
        try:
            for url in urls:
                if crawled_count >= self.max_pages:
                    logger.info(f"Reached maximum page limit: {self.max_pages}")
                    break
                
                # Rate limiting
                await self._rate_limit_delay()
                
                # Crawl single URL
                doc = await self._crawl_url(url)
                if doc:
                    documents.append(doc)
                    crawled_count += 1
                
                # If crawling with depth, get linked pages (temporarily disabled for testing)
                # if self.max_depth > 1:
                #     linked_docs = await self._crawl_with_depth(url, depth=1)
                #     documents.extend(linked_docs[:self.max_pages - crawled_count])
                #     crawled_count += len(linked_docs)
            
            logger.info(f"Crawled {len(documents)} web pages")
            return documents
            
        except Exception as e:
            logger.error(f"Error crawling web content: {e}")
            return documents
    
    async def _rate_limit_delay(self):
        """Apply rate limiting between requests"""
        if self.rate_limit > 0:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            min_delay = 1.0 / self.rate_limit
            
            if time_since_last < min_delay:
                delay = min_delay - time_since_last
                await asyncio.sleep(delay)
            
            self._last_request_time = time.time()
    
    async def _crawl_url(self, url: str) -> Optional[Document]:
        """Crawl a single URL and return Document"""
        try:
            logger.debug(f"Crawling URL: {url}")
            
            result = await self.crawler.arun(
                url=url,
                word_count_threshold=10,
                bypass_cache=True,
                timeout=self.timeout
            )
            
            if not result.success:
                logger.warning(f"Failed to crawl {url}: {result.error_message}")
                return None
            
            if not result.cleaned_html or len(result.cleaned_html.strip()) < 50:
                logger.warning(f"No meaningful content found for {url}")
                return None
            
            # Create document metadata
            metadata = DocumentMetadata(
                source="web_crawl",
                url=url,
                title=result.metadata.get('title', ''),
                file_name=urlparse(url).path.split('/')[-1] or 'index.html',
                file_type='.html',
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata={
                    'status_code': result.status_code,
                    'content_length': len(result.cleaned_html),
                    'links_found': len(result.links.get('internal', [])) + len(result.links.get('external', [])),
                    'media_found': len(result.media.get('images', [])),
                    'crawl_timestamp': result.metadata.get('timestamp', ''),
                }
            )
            
            # Create document
            document = Document(
                id=str(hash(f"{url}_{datetime.now().isoformat()}")),
                content=result.cleaned_html,
                metadata=metadata
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error crawling URL {url}: {e}")
            return None
    
    async def _crawl_with_depth(self, base_url: str, depth: int) -> List[Document]:
        """Crawl linked pages up to specified depth"""
        if depth >= self.max_depth:
            return []
        
        documents = []
        
        try:
            # Get the base page first to extract links
            result = await self.crawler.arun(url=base_url)
            
            if not result.success:
                return []
            
            # Extract internal links
            internal_links = result.links.get('internal', [])
            
            # Limit the number of links to crawl
            links_to_crawl = internal_links[:min(5, self.max_pages)]
            
            for link in links_to_crawl:
                # Convert relative URLs to absolute
                if not isinstance(link, str):
                    continue
                full_url = urljoin(base_url, link)
                
                # Rate limiting
                await self._rate_limit_delay()
                
                # Crawl the linked page
                doc = await self._crawl_url(full_url)
                if doc:
                    documents.append(doc)
                
                # Recursive crawling
                if depth + 1 < self.max_depth:
                    deeper_docs = await self._crawl_with_depth(full_url, depth + 1)
                    documents.extend(deeper_docs)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in depth crawling from {base_url}: {e}")
            return []
    
    async def crawl_sitemap(self, sitemap_url: str) -> List[Document]:
        """Crawl URLs from a sitemap"""
        try:
            # This is a placeholder - you'd implement sitemap parsing here
            logger.info(f"Sitemap crawling not yet implemented for {sitemap_url}")
            return []
            
        except Exception as e:
            logger.error(f"Error crawling sitemap {sitemap_url}: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of web crawler"""
        try:
            my_port = os.getenv('API_PORT', '8000')
            test_url = f"http://localhost:{my_port}"
            result = await self.crawler.arun(url=test_url, timeout=10)
            
            if result.success:
                return {
                    "status": "healthy",
                    "crawler_connected": True,
                    "rate_limit": self.rate_limit,
                    "max_pages": self.max_pages,
                    "max_depth": self.max_depth,
                    "test_crawl": "successful"
                }
            else:
                return {
                    "status": "degraded",
                    "message": f"Test crawl failed: {result.error_message}",
                    "crawler_connected": True
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}"
            }
    
    def get_connector_info(self) -> Dict[str, Any]:
        """Return connector metadata"""
        return {
            "name": "WebConnector",
            "type": "web_crawler",
            "status": "connected" if self._connected else "disconnected",
            "rate_limit": self.rate_limit,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "user_agent": self.user_agent,
            "respect_robots": self.respect_robots
        }