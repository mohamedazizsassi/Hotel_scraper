"""
hotel_scraper/constants.py

Static configuration: city mappings, scraping parameters.
"""

# Stay durations to scrape
NIGHTS_OPTIONS = (1, 3, 5)

# ------------------------------------------------------------------ #
#  City mappings per source                                            #
# ------------------------------------------------------------------ #

PROMOHOTEL_CITIES = {
    18: "ain-draham",
    12: "bizerte",
    1:  "djerba",
    17: "douz",
    2:  "hammamet",
    26: "kairouan",
    24: "kasserine",
    16: "kelibia",
    4:  "mahdia",
    8:  "monastir",
    9:  "nabeul",
    13: "sfax",
    3:  "sousse",
    7:  "tabarka",
    15: "tataouine",
    11: "tozeur",
    6:  "tunis",
    5:  "tunis-gammarth",
    10: "zarzis",
}

TUNISIEPROMO_CITIES = {
    31:   "ain-draham",
    48:   "bizerte",
    18:   "djerba",
    20:   "douz",
    55:   "gabes",
    54:   "gafsa",
    10:   "hammamet",
    17:   "kairouan",
    12:   "kelibia",
    35:   "mahdia",
    37:   "monastir",
    11:   "nabeul",
    39:   "sfax",
    34:   "sousse",
    33:   "tabarka",
    70:   "tataouine",
    47:   "tozeur",
    6479: "tunis",
    30:   "tunis-gammarth",
    19:   "zarzis",
}
