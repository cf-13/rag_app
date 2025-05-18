import requests
from bs4 import BeautifulSoup
import json
import os
import time
import random
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Set

class HelpPageCrawler:
    def __init__(self, base_url: str, output_dir: str):
        self.base_url = base_url
        self.output_dir = output_dir
        self.visited_urls: Set[str] = set()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a session to maintain cookies
        self.session = requests.Session()
        
        # Set more realistic headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Ch-Ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL to avoid duplicates."""
        parsed = urlparse(url)
        # Remove fragments
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{parsed.query}"
    
    def crawl(self, start_url: str, max_pages: int = 100) -> List[Dict[str, Any]]:
        """Crawl help pages starting from given URL."""
        queue = [start_url]
        documents = []
        
        # First try to access the homepage to get cookies
        try:
            print(f"Accessing homepage to establish session...")
            self.session.get(self.base_url, headers=self.headers, timeout=15)
            time.sleep(3)  # Wait before proceeding
        except Exception as e:
            print(f"Error accessing homepage: {e}")
        
        while queue and len(documents) < max_pages:
            url = queue.pop(0)
            normalized_url = self.normalize_url(url)
            
            if normalized_url in self.visited_urls:
                continue
                
            self.visited_urls.add(normalized_url)
            print(f"Crawling: {url}")
            
            try:
                # Add referer to make the request look more natural
                current_headers = self.headers.copy()
                if len(self.visited_urls) > 1:
                    current_headers['Referer'] = list(self.visited_urls)[-2]
                
                # Add random delay between requests to avoid being flagged as a bot
                time.sleep(2 + random.random() * 3)  # Random delay between 2-5 seconds
                
                response = self.session.get(url, headers=current_headers, timeout=15)
                
                # Handle different status codes
                if response.status_code == 403:
                    print(f"Access forbidden (403) for {url} - trying with different approach")
                    # Try with a different user agent
                    alternate_headers = current_headers.copy()
                    alternate_headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
                    time.sleep(5)  # Longer wait
                    response = self.session.get(url, headers=alternate_headers, timeout=15)
                
                if response.status_code != 200:
                    print(f"Failed to fetch {url}: {response.status_code}")
                    continue
                
                # Successful response
                print(f"Successfully accessed {url}")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Debug: Save HTML to examine structure
                if len(documents) == 0:  # Save first page for debugging
                    with open(os.path.join(self.output_dir, "first_page.html"), "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"Saved first page HTML for debugging")
                
                # Extract content
                content = self._extract_content(soup)
                if not content or not content.get('title') or not content.get('text'):
                    print(f"Could not extract content from {url}")
                    # Try to understand page structure
                    title_candidates = soup.find_all(['h1', 'h2', 'title'])
                    if title_candidates:
                        print(f"Possible title elements found: {[t.get_text(strip=True) for t in title_candidates[:3]]}")
                    continue
                
                # Add URL to content
                content['url'] = url
                documents.append(content)
                
                # Save document
                self._save_document(content)
                
                # Find links to other help pages
                new_urls = self._find_help_page_links(soup, url)
                print(f"Found {len(new_urls)} new links on {url}")
                for new_url in new_urls:
                    if self.normalize_url(new_url) not in self.visited_urls:
                        queue.append(new_url)
                        
            except Exception as e:
                print(f"Error crawling {url}: {e}")
        
        return documents
    
    def _extract_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract relevant content from the help page."""
        try:
            # For Python docs specifically
            title_elem = None
            content_elem = None
            
            # Python docs specific selectors
            title_elem = soup.select_one('h1')
            content_elem = soup.select_one('div.body')
            
            # If not found with Python docs selectors, try generic selectors
            if not title_elem:
                for selector in ['h1.article-title', 'h1.article_title', 'h1.page-header', 'h1', '.article-title', '.heading-title']:
                    title_elem = soup.select_one(selector)
                    if title_elem and title_elem.get_text(strip=True):
                        break
            
            if not content_elem:
                for selector in ['div.body', 'div.document', 'div.article-body', '.article_body', '.article-content', '.content-body', 'main article', '.main-content']:
                    content_elem = soup.select_one(selector)
                    if content_elem and content_elem.get_text(strip=True):
                        break
            
            # If still not found, try more generic approach
            if not title_elem:
                title_elem = soup.find(['h1', 'h2'])
            
            if not content_elem:
                content_elem = soup.find(['main', 'article', 'div.content'])
            
            if not title_elem or not content_elem:
                return {}
            
            title = title_elem.get_text(strip=True)
            
            # Extract text and clean it up
            paragraphs = content_elem.find_all(['p', 'li', 'h2', 'h3', 'h4', 'pre', 'code'])
            text = "\n\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            if not text:
                text = content_elem.get_text(strip=True)
            
            # Try to extract categories from breadcrumbs
            categories = []
            rel_nav = soup.select_one('div.related')
            if rel_nav:
                links = rel_nav.find_all('a')
                categories = [link.get_text(strip=True) for link in links if link.get_text(strip=True)]
            
            # Try to find last updated date
            last_updated = None
            last_updated_elem = soup.select_one('div.footer')
            if last_updated_elem:
                last_updated = last_updated_elem.get_text(strip=True)
            
            return {
                'title': title,
                'text': text,
                'categories': categories,
                'metadata': {
                    'last_updated': last_updated,
                }
            }
        except Exception as e:
            print(f"Error extracting content: {e}")
            return {}
    
    def _find_help_page_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Find links to other help pages."""
        links = []
        # Find all links in the page
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(current_url, href)
            
            # Only follow links to the same domain and path
            parsed_url = urlparse(full_url)
            base_domain = urlparse(self.base_url).netloc
            base_path = urlparse(self.base_url).path
            
            # For Python docs, stay within the tutorial section
            if parsed_url.netloc == base_domain and parsed_url.path.startswith(base_path):
                # Avoid links to indices, TOCs, etc.
                if any(x in parsed_url.path for x in ['genindex', 'modindex', 'search', 'glossary']):
                    continue
                    
                # Avoid links to non-HTML content or non-Python docs
                if not parsed_url.path.endswith('.html') and '.' in parsed_url.path:
                    continue
                
                # Prioritize links with content keywords
                link_text = a_tag.get_text(strip=True).lower()
                tutorial_terms = ['チュートリアル', 'tutorial', 'guide', 'ガイド', '入門', 'introduction', 'basics', '基本']
                
                # Add tutorial links first, other links after
                if any(term in link_text for term in tutorial_terms):
                    links.insert(0, full_url)  # Add to beginning of list
                else:
                    links.append(full_url)  # Add to end of list
        
        return links
    
    def _save_document(self, document: Dict[str, Any]) -> None:
        """Save the document to disk."""
        # Create a filename from the URL
        parsed_url = urlparse(document['url'])
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) > 0:
            # Use the last part of the path as filename
            filename = f"{path_parts[-1]}.json"
        else:
            # Fallback for homepage or root pages
            filename = f"{parsed_url.netloc.replace('.', '_')}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        
        print(f"Saved: {file_path}")


def main():
    # Using a public help page for demonstration
    base_url = "https://docs.python.org/ja/3/tutorial/"
    start_url = base_url  # Start from the Python tutorial page
    output_dir = "crawled_data"
    
    crawler = HelpPageCrawler(base_url, output_dir)
    documents = crawler.crawl(start_url, max_pages=10)  # Limit to 10 pages for testing
    
    print(f"Crawled {len(documents)} documents")

if __name__ == "__main__":
    main()