import scrapy


class HotelPriceItem(scrapy.Item):
    """
    Represents a single scraped room price record.
    One item = one room option for one hotel on one check-in date.
    """
    # Source metadata
    source         = scrapy.Field()   # "promohotel" | "tunisiepromo"
    scraped_at     = scrapy.Field()   # ISO datetime of scrape
    scrape_run_id  = scrapy.Field()   # groups all items from one scheduled run

    # Search parameters
    check_in           = scrapy.Field()   # "YYYY-MM-DD"
    check_out          = scrapy.Field()   # "YYYY-MM-DD"
    nights             = scrapy.Field()   # int
    days_until_checkin = scrapy.Field()   # int, (check_in - today).days
    city_id            = scrapy.Field()   # int
    adults             = scrapy.Field()   # int
    children           = scrapy.Field()   # int

    # Hotel info
    hotel_name            = scrapy.Field()
    hotel_name_normalized = scrapy.Field()   # lowercased, accent-stripped, prefix-removed
    city_name             = scrapy.Field()   # human-readable city name
    stars                 = scrapy.Field()

    # Boarding / room
    boarding_name  = scrapy.Field()   # e.g. "Demi-pension"
    room_name      = scrapy.Field()
    price          = scrapy.Field()   # float, TND (total for stay)
    price_per_night = scrapy.Field()  # float, TND (price / nights)
    sur_demande    = scrapy.Field()   # bool (promohotel only)

    # Supplements (list of {name, price})
    supplements    = scrapy.Field()
