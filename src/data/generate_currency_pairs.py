def create_cross_pairs(currencies):
    cross_pairs = []
    for currency in currencies:
        for other_currency in currencies:
            if other_currency != currency:
                pair = f"{currency}{other_currency}"
                if pair not in cross_pairs:
                    cross_pairs.append(pair)
    return cross_pairs


def create_currency_currencies_pair(currency, currencies):
    pairs = []
    for other_currency in currencies:
        pair1 = f"{currency}{other_currency}"
        pair2 = f"{other_currency}{currency}"
        
        pairs.append(pair1)
        pairs.append(pair2)

    return pairs


MAJOR_CURRENCIES = [
    "USD",
    "EUR",
    "JPY",
    "GBP",
    "CHF",
    "CAD",
    "AUD",
    "NZD"
]

if __name__ == "__main__":
    MINOR_CURRENCIES = list(filter(
        lambda currency: currency not in MAJOR_CURRENCIES, 
        open("../data/groups/currencies.txt", "r").read().split("\n")
    ))


    MAJOR_PAIRS = create_cross_pairs(MAJOR_CURRENCIES)

    MAJOR_MINOR_PAIRS = []
    for currency in MAJOR_CURRENCIES:
        MAJOR_MINOR_PAIRS += create_currency_currencies_pair(currency, MINOR_CURRENCIES)


    MINOR_MINOR_PAIRS = create_cross_pairs(MINOR_CURRENCIES)


    COMBINED_PAIRS = MAJOR_PAIRS + MAJOR_MINOR_PAIRS + MINOR_MINOR_PAIRS

    open("../data/groups/mega_pairs.txt", "w").write("\n".join(COMBINED_PAIRS))