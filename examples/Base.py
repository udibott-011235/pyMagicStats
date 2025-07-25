# scraper/base.py
from __future__ import annotations

import abc
import re
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


class SuperScraper(abc.ABC):
    """
    Clase base para todos los supermercados.
    Implementa la lógica común:
      - construir URL a partir de EAN
      - hacer GET con requests
      - devolver dict estandarizado
    """

    def __init__(self, ean: str, timeout: int = 10) -> None:
        self.ean = ean.strip()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ---------- API pública ----------
    def scrape(self) -> Dict[str, Optional[str]]:
        url = self.build_url()
        html = self._download(url)
        soup = BeautifulSoup(html, "html.parser")
        return self.parse_product_page(soup, url)

    # ---------- Métodos a sobreescribir ----------
    @abc.abstractmethod
    def build_url(self) -> str:
        """Devuelve la URL de producto a partir del EAN"""
        raise NotImplementedError

    @abc.abstractmethod
    def parse_product_page(
        self, soup: BeautifulSoup, url: str
    ) -> Dict[str, Optional[str]]:
        """Extrae los campos requeridos del HTML"""
        raise NotImplementedError

    # ---------- Helper común ----------
    def _download(self, url: str) -> str:
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    # ---------- Utilidades genéricas ----------
    @staticmethod
    def _get_meta_property(soup: BeautifulSoup, prop: str) -> Optional[str]:
        tag = soup.find("meta", {"property": prop})
        return tag["content"].strip() if tag and tag.get("content") else None
