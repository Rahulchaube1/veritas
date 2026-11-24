"""A small controlled grammar for exact counts. Entity names retain their accents."""

import re
from dataclasses import dataclass

KINDS = {
    "station": "station",
    "estación": "station",
    "depot": "depot",
    "depósito": "depot",
    "hospital": "hospital",
    "library": "library",
    "biblioteca": "library",
    "airport": "airport",
    "aeropuerto": "airport",
    "museum": "museum",
    "museo": "museum",
    "port": "port",
    "puerto": "port",
    "hotel": "hotel",
}
UNITS = {
    "berth": "berth",
    "berths": "berth",
    "atracadero": "berth",
    "atracaderos": "berth",
    "bed": "bed",
    "beds": "bed",
    "cama": "bed",
    "camas": "bed",
    "room": "room",
    "rooms": "room",
    "habitación": "room",
    "habitaciones": "room",
    "seat": "seat",
    "seats": "seat",
    "asiento": "seat",
    "asientos": "seat",
    "runway": "runway",
    "runways": "runway",
    "pista de aterrizaje": "runway",
    "pistas de aterrizaje": "runway",
    "book": "book",
    "books": "book",
    "libro": "book",
    "libros": "book",
}
NAME = r"[\wÀ-ÿ'-]+(?: [\wÀ-ÿ'-]+){0,5}"
SITE = re.compile(rf"(?:(?:the|el|la) )?(\w+) ({NAME})", re.I)
COUNT = re.compile(r"(.+?) (?:has|tiene) ([0-9]{1,12}) ([\w ]{1,30})", re.I)


@dataclass(frozen=True)
class Site:
    kind: str
    name: str


@dataclass(frozen=True)
class Count:
    site: Site
    number: int
    unit: str


def statement(text: str) -> str:
    return " ".join(text.strip().removesuffix(".").split())


def parse_site(text: str) -> Site | None:
    m = SITE.fullmatch(text)
    if not m or m[1].casefold() not in KINDS:
        return None
    return Site(KINDS[m[1].casefold()], m[2].casefold())


def parse_count(text: str) -> Count | None:
    m = COUNT.fullmatch(statement(text))
    if not m or (unit := UNITS.get(m[3].casefold())) is None:
        return None
    site = parse_site(m[1])
    return Count(site, int(m[2]), unit) if site else None
