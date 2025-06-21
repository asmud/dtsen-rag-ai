import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse
import time
import re

import httpx
from bs4 import BeautifulSoup

from connectors.base_connector import BaseConnector
from models.document import Document, DocumentMetadata

logger = logging.getLogger(__name__)

class WebConnector(BaseConnector):
    """Connector for web content using httpx and BeautifulSoup"""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.rate_limit = self.settings.CRAWL_RATE_LIMIT
        self.timeout = self.settings.CRAWL_TIMEOUT
        self.max_pages = self.settings.CRAWL_MAX_PAGES
        self.user_agent = self.settings.CRAWL_USER_AGENT
        self.respect_robots = self.settings.CRAWL_RESPECT_ROBOTS
        self.max_depth = self.settings.CRAWL_MAX_DEPTH
        self._connected = False
        self._last_request_time = 0
    
    async def connect(self) -> bool:
        """Initialize the HTTP client"""
        try:
            # Close existing client if it exists
            if self.client:
                try:
                    await self.client.aclose()
                except:
                    pass
            
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    'User-Agent': self.user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=100)
            )
            self._connected = True
            logger.info("Connected to web client")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to web client: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Close the HTTP client"""
        try:
            if self.client:
                await self.client.aclose()
            self._connected = False
            logger.info("Disconnected from web client")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting web client: {e}")
            return False
    
    async def fetch_data(self, **kwargs) -> List[Document]:
        """Fetch web content from URLs"""
        urls = kwargs.get('urls', [])
        if not urls:
            logger.warning("No URLs provided for web crawling")
            return []
        
        documents = []
        crawled_count = 0
        max_retries = 3
        
        # Ensure connection
        if not self._connected:
            await self.connect()
        
        for url in urls:
            if crawled_count >= self.max_pages:
                logger.info(f"Reached maximum page limit: {self.max_pages}")
                break
            
            # Retry mechanism for connection issues
            for attempt in range(max_retries):
                try:
                    # Rate limiting
                    await self._rate_limit_delay()
                    
                    # Crawl single URL
                    doc = await self._crawl_url(url)
                    if doc:
                        documents.append(doc)
                        crawled_count += 1
                    break  # Success, break retry loop
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    # Check for connection errors
                    if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network']):
                        logger.warning(f"Connection error on attempt {attempt + 1} for {url}: {e}")
                        if attempt < max_retries - 1:
                            # Wait before retry
                            await asyncio.sleep(2 ** attempt)
                            continue
                    else:
                        logger.error(f"Error crawling {url}: {e}")
                        break  # Non-connection error, don't retry
        
        logger.info(f"Crawled {len(documents)} web pages")
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
            
            if not self.client:
                logger.error(f"HTTP client not initialized for {url}")
                return None
            
            # Make HTTP request
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else ''
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            # Extract main content
            content_text = self._extract_main_content(soup)
            
            if len(content_text.strip()) < 50:
                logger.warning(f"No meaningful content found for {url}")
                return None
            
            # Extract links for metadata
            links = soup.find_all('a', href=True)
            internal_links = []
            external_links = []
            
            for link in links:
                href = link['href']
                if href.startswith('http'):
                    if urlparse(url).netloc in href:
                        internal_links.append(href)
                    else:
                        external_links.append(href)
                elif href.startswith('/'):
                    internal_links.append(urljoin(url, href))
            
            # Extract images for metadata
            images = soup.find_all('img', src=True)
            
            # Create document metadata
            metadata = DocumentMetadata(
                source="web_crawl",
                url=url,
                title=title,
                file_name=urlparse(url).path.split('/')[-1] or 'index.html',
                file_type='.html',
                created_at=datetime.now(),
                processed_at=datetime.now(),
                extra_metadata={
                    'status_code': response.status_code,
                    'content_length': len(content_text),
                    'links_found': len(internal_links) + len(external_links),
                    'internal_links': len(internal_links),
                    'external_links': len(external_links),
                    'media_found': len(images),
                    'crawl_timestamp': datetime.now().isoformat(),
                    'content_type': response.headers.get('content-type', ''),
                }
            )
            
            # Create document
            document = Document(
                id=str(hash(f"{url}_{datetime.now().isoformat()}")),
                content=content_text,
                metadata=metadata
            )
            
            return document
            
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error {e.response.status_code} for {url}: {e}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error for {url}: {e}")
            raise  # Re-raise to trigger retry logic
        except Exception as e:
            logger.error(f"Error crawling URL {url}: {e}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from BeautifulSoup object"""
        # Try to find main content areas first
        main_selectors = [
            'main', 'article', '.content', '#content', 
            '.main-content', '#main-content', '.post-content',
            '.entry-content', '.article-content'
        ]
        
        main_content = None
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text and clean it up
        text = main_content.get_text()
        
        # Clean up the text
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) > 3:  # Skip very short lines
                lines.append(line)
        
        # Join lines and clean up extra whitespace
        content = '\n'.join(lines)
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Remove excessive newlines
        content = re.sub(r' +', ' ', content)  # Remove excessive spaces
        
        return content.strip()
    
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
        """Check health of web client"""
        try:
            # Check basic connection first
            if not self._connected or not self.client:
                await self.connect()
            
            if not self._connected:
                return {
                    "status": "unhealthy",
                    "message": "Cannot establish client connection",
                    "crawler_connected": False
                }
            
            # Simple test - try to fetch a basic page
            test_url = "https://httpbin.org/robots.txt"
            response = await self.client.get(test_url, timeout=10)
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "crawler_connected": True,
                "rate_limit": self.rate_limit,
                "max_pages": self.max_pages,
                "max_depth": self.max_depth,
                "test_crawl": "successful"
            }
                
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network']):
                self._connected = False
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {e}",
                "crawler_connected": self._connected
            }
    
    def get_connector_info(self) -> Dict[str, Any]:
        """Return connector metadata"""
        return {
            "name": "WebConnector",
            "type": "web_scraper",
            "status": "connected" if self._connected else "disconnected",
            "rate_limit": self.rate_limit,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "user_agent": self.user_agent,
            "respect_robots": self.respect_robots,
            "method": "httpx + BeautifulSoup"
        }