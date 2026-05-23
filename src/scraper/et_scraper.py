"""
Economic Times news scraper using Selenium (headless Chrome).
Returns a DataFrame with columns: ['title', 'link']
"""

import time
import pandas as pd


def scrape_et_news(max_steps: int = 60, sleep_s: float = 1.0,
                   headless: bool = True, page_timeout: int = 20) -> pd.DataFrame:
    """
    Scrape financial headlines from Economic Times Markets section.

    Args:
        max_steps: Maximum scroll/click iterations.
        sleep_s: Sleep time (seconds) between scroll steps.
        headless: Run Chrome in headless mode.
        page_timeout: Page and script load timeout in seconds.

    Returns:
        pd.DataFrame with columns ['title', 'link'], or empty DataFrame on failure.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from webdriver_manager.chrome import ChromeDriverManager
        from bs4 import BeautifulSoup
    except Exception as e:
        print(f"[ET Scraper] Selenium stack unavailable: {e}")
        return pd.DataFrame(columns=["title", "link"])

    URL = "https://economictimes.indiatimes.com/markets/stocks/news/"
    t0 = time.time()
    titles_links = []

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    try:
        opts.page_load_strategy = "eager"
    except Exception:
        pass

    driver = None
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)
        driver.set_page_load_timeout(page_timeout)
        driver.set_script_timeout(page_timeout)
        driver.get(URL)

        try:
            WebDriverWait(driver, 15).until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.eachStory")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, "article")),
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/markets/']")),
                )
            )
        except Exception:
            pass

        def _try_click_load_more():
            selectors = [
                ".autoload_continue", ".loadMore", ".load_more", ".moreBox",
                "button[aria-label*='more' i]", "a[aria-label*='more' i]",
            ]
            for sel in selectors:
                for el in driver.find_elements(By.CSS_SELECTOR, sel):
                    try:
                        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                        time.sleep(0.4)
                        el.click()
                        return True
                    except Exception:
                        continue
            return False

        same_count = 0
        last_h = driver.execute_script("return document.body.scrollHeight;")
        for _ in range(max_steps):
            try:
                clicked = _try_click_load_more()
            except Exception:
                clicked = False
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(sleep_s)
            h = driver.execute_script("return document.body.scrollHeight;")
            same_count = same_count + 1 if (h <= last_h and not clicked) else 0
            last_h = h
            if same_count >= 3:
                break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        cand = (
            soup.select("div.eachStory a")
            + soup.select("article a")
            + soup.select("a[href*='/markets/']")
        )

        seen = set()
        for a in cand:
            href = (a.get("href") or "").strip()
            txt = (a.get_text(" ", strip=True) or "").strip()
            if not href or not txt:
                continue
            if href.startswith("/"):
                href = "https://economictimes.indiatimes.com" + href
            key = (txt, href)
            if key in seen:
                continue
            seen.add(key)
            if len(txt) >= 25 and "/markets/" in href and "javascript:" not in href:
                titles_links.append({"title": txt, "link": href})

        df_out = pd.DataFrame(titles_links).drop_duplicates().reset_index(drop=True)
        if len(df_out) > 500:
            df_out = df_out.head(500)

        print(f"[ET Scraper] {len(df_out)} articles | {time.time() - t0:.1f}s")
        return df_out

    except Exception as e:
        print(f"[ET Scraper] Failed: {e}")
        return pd.DataFrame(columns=["title", "link"])

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
