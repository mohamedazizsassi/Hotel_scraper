import scrapy


class HotelPriceItem(scrapy.Item):
    """
    Represents a single scraped room price record.
    One item = one room option for one hotel on one check-in date.
    """
    # Source metadata
    source         = scrapy.Field()   # "promohotel" | "tunisiepromo"
    scraped_at     = scrapy.Field()   # ISO datetime of scrape

    # Search parameters
    check_in       = scrapy.Field()   # "YYYY-MM-DD"
    check_out      = scrapy.Field()   # "YYYY-MM-DD"
    city_id        = scrapy.Field()   # int
    adults         = scrapy.Field()   # int

    # Hotel info
    hotel_name     = scrapy.Field()
    stars          = scrapy.Field()

    # Boarding / room
    boarding_name  = scrapy.Field()   # e.g. "Demi-pension"
    room_name      = scrapy.Field()
    price          = scrapy.Field()   # float, TND
    sur_demande    = scrapy.Field()   # bool (promohotel only)

    # Supplements (list of {name, price})
    supplements    = scrapy.Field()
