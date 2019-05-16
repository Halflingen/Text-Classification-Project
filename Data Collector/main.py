import tv2_football_scraper
import vg_football_scraper
import time

def main():
    tv2_urls = tv2_football_scraper.get_valid_hrefs()
    vg_urls = vg_football_scraper.get_valid_hrefs()

    for i in range(500):
        vg_football_scraper.get_articles(vg_urls[i])
        tv2_football_scraper.get_articles(tv2_urls[i])
        time.sleep(10)

main()
