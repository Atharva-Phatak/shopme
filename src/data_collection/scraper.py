import argparse
import math
import os
import re
import time

import boto3
import pandas as pd
import requests
from bs4 import BeautifulSoup

from categories import CATEGORIES


headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
}


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket-path",
        help="root directory to store metadata",
    )
    args = parser.parse_args()
    return args


def get_pagination(url: str) -> int:
    """Paginate pages and calculate the total number of products"""
    page = requests.get(url=url, headers=headers)
    soupObject = BeautifulSoup(
        page.text, "html.parser"
    )
    total_items = (
        soupObject.find(
            "div", {"class": "filter-pagination"}
        )
        .text.strip()
        .split()[0]
    )
    time.sleep(20)
    return total_items


def get_product_links(
    soupObject: BeautifulSoup,
) -> list:
    """Get link of products"""
    links = soupObject.find_all(
        "a", {"class": "link"}
    )
    links = [
        link["href"]
        for link in links
        if "productpage" in link["href"]
    ]
    return links


def get_product_ids(links: list) -> list:
    """Get product ids"""
    ids = []
    for link in links:
        idx = re.findall(
            pattern=f"\d+", string=link
        )
        ids.append(int(*idx))
    return ids


def get_product_prices(
    soupObject: BeautifulSoup,
) -> list:
    """Get price of each product"""
    prices = soupObject.find_all(
        "span", {"class": "price regular"}
    )
    prices = [price.text for price in prices]
    prices = [
        float(price.replace("$", ""))
        for price in prices
    ]
    return prices


def get_image_links(
    soupObject: BeautifulSoup,
) -> list:
    """Get links of product images."""
    atags = soupObject.find_all(
        "img", {"class": "item-image"}
    )
    image_links = [
        x["data-altimage"] for x in atags
    ]
    image_links = [
        x.replace("/style", "/main")
        for x in image_links
    ]
    return image_links


def build_pages(
    gender, product_category: str
) -> dict:
    """Get pages and scrape all the necessary metadata"""
    print(f"Scraping : {product_category}")
    url = f"https://www2.hm.com/en_ca/{gender}/shop-by-product/{product_category}.html?offset=0&page-size={2}"
    total_items = get_pagination(url=url)
    print(
        f"{product_category} : {total_items} items"
    )
    url = f"https://www2.hm.com/en_ca/{gender}/shop-by-product/{product_category}.html?offset=0&page-size={total_items}"
    data = {}
    try:
        page = requests.get(url, headers=headers)
        if page.status_code == 200:
            soup = BeautifulSoup(
                page.text, "html.parser"
            )
            prices = get_product_prices(
                soupObject=soup
            )
            links = get_product_links(
                soupObject=soup
            )
            ids = get_product_ids(links=links)
            image_links = get_image_links(
                soupObject=soup
            )
            data = {
                "product_prices": prices,
                "product_page_links": links,
                "product_ids": ids,
                "image_links": image_links,
            }
            print(
                f"Successfully Scraped : {product_category}"
            )
            print("====" * 5)
    except:
        pass

    return data


def collect_products(
    gender: str,
    categories: list,
    bucket_name: str,
):
    """Method to collect all the product data for particular category and gender and finally push to an s3 bucket"""
    for category in categories:
        data = build_pages(category)
        if data == {}:
            print(f"Could not scrape: {category}")
        else:
            frame = pd.DataFrame.from_dict(data)
            # Saving to s3 bucket
            frame.to_csv(
                f"s3://{bucket_name}/{gender}/{category}.csv",
                index=False,
            )
        time.sleep(15)


if __name__ == "__main__":
    args = create_parser()
    for gender, categories in CATEGORIES.items():
        collect_products(
            gender=gender,
            categories=categories,
            path=args.bucket_path,
        )
