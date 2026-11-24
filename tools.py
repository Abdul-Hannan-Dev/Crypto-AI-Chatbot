
import os
import json
import difflib
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from functools import lru_cache

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent / "kb.json"
API_URL = "https://api.freecryptoapi.com/v1/getData"
COIN_LIST_URL = "https://api.coingecko.com/api/v3/coins/list"

API_KEY = os.getenv("CRYPTO_API_KEY")

KB_LOCK = threading.Lock()

# Reject quotes older than this for current-price requests.
MAX_QUOTE_AGE_SECONDS = 1800


def normalize(value):
    return str(value or "").strip().lower()


def parse_timestamp(value):
    if not value:
        return None

    try:
        timestamp = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(timestamp)

        # Assume UTC+3 for API timestamps without timezone
        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone(timedelta(hours=3))
            )

        return parsed.astimezone(timezone.utc)

    except (ValueError, TypeError):
        return None


def load_kb():
    if not KB_PATH.exists():
        return {
            "metadata": {},
            "coins": []
        }

    with open(KB_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Invalid KB structure")

    if not isinstance(data.get("coins"), list):
        raise ValueError("KB coins must be a list")

    return data


def save_kb(data):
    temp_path = KB_PATH.with_suffix(".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        os.replace(temp_path, KB_PATH)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def get_static_records(data):
    return [
        coin for coin in data["coins"]
        if isinstance(coin, dict) and coin.get("coin")
    ]


def get_snapshot_records(data):
    return [
        coin for coin in data["coins"]
        if (
            isinstance(coin, dict)
            and not coin.get("coin")
            and coin.get("symbol")
            and coin.get("last") is not None
        )
    ]


def find_static_coin(query, data):
    query = normalize(query)
    coins = get_static_records(data)

    # Exact matching
    matches = [
        coin for coin in coins
        if query in (
            normalize(coin.get("coin")),
            normalize(coin.get("symbol"))
        )
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        return None

    # Minor spelling corrections for coin names
    names = {
        normalize(coin["coin"]): coin
        for coin in coins
    }

    close_matches = difflib.get_close_matches(
        query,
        names.keys(),
        n=2,
        cutoff=0.85
    )

    if len(close_matches) == 1:
        return names[close_matches[0]]

    return None


def get_latest_snapshot(symbol, data):
    symbol = normalize(symbol)

    snapshots = [
        coin for coin in get_snapshot_records(data)
        if normalize(coin.get("symbol")) == symbol
    ]

    if not snapshots:
        return None

    snapshots.sort(
        key=lambda coin: (
            parse_timestamp(coin.get("date"))
            or datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True
    )

    return snapshots[0]


def get_coin_directory():
    response = requests.get(
        COIN_LIST_URL,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise ValueError("Invalid coin directory")

    return data


@lru_cache(maxsize=1)
def cached_coin_directory():
    return get_coin_directory()


def resolve_coin(query):
    """
    Resolve an exact name, symbol, or minor typo.

    Returns a canonical coin name and symbol.
    Ambiguous symbols are rejected.
    """

    query = normalize(query)

    if not query:
        return None

    data = load_kb()

    local_coin = find_static_coin(query, data)

    if local_coin:
        return {
            "name": local_coin["coin"],
            "symbol": local_coin["symbol"].upper()
        }

    # Check known snapshots for a unique exact symbol
    snapshot_symbols = {
        normalize(coin.get("symbol"))
        for coin in get_snapshot_records(data)
    }

    if query in snapshot_symbols:
        # Directory validation is preferred when available.
        pass

    try:
        directory = cached_coin_directory()

        name_matches = [
            coin for coin in directory
            if query in (
                normalize(coin.get("name")),
                normalize(coin.get("id"))
            )
        ]

        if len(name_matches) == 1:
            coin = name_matches[0]

            return {
                "name": coin["name"],
                "symbol": coin["symbol"].upper()
            }

        if len(name_matches) > 1:
            return None

        symbol_matches = [
            coin for coin in directory
            if normalize(coin.get("symbol")) == query
        ]

        if len(symbol_matches) == 1:
            coin = symbol_matches[0]

            return {
                "name": coin["name"],
                "symbol": coin["symbol"].upper()
            }

        if len(symbol_matches) > 1:
            return None

        # Fuzzy matching is limited to full coin names
        names = {
            normalize(coin.get("name")): coin
            for coin in directory
            if coin.get("name")
        }

        matches = difflib.get_close_matches(
            query,
            names.keys(),
            n=2,
            cutoff=0.90
        )

        if len(matches) == 1:
            coin = names[matches[0]]

            return {
                "name": coin["name"],
                "symbol": coin["symbol"].upper()
            }

    except requests.exceptions.RequestException:
        logger.warning("Coin directory unavailable")

    except Exception:
        logger.exception("Coin resolution failed")

    # Offline fallback for symbols already in the KB
    if query in snapshot_symbols:
        return {
            "name": query.upper(),
            "symbol": query.upper()
        }

    return None


def update_market_cache(symbol, market_data):
    """
    Preserve static facts while updating the market cache.

    Remove duplicate historical API-only records for
    the same symbol after successfully saving the quote.
    """

    with KB_LOCK:
        data = load_kb()
        symbol = symbol.upper()

        timestamp = market_data["date"]

        new_cache = {
            "last_price": float(market_data["last"]),
            "currency": "USD",
            "price_timestamp": timestamp,
            "daily_change_percentage": market_data.get(
                "daily_change_percentage"
            ),
            "source_exchange": market_data.get(
                "source_exchange"
            ),
            "lowest": market_data.get("lowest"),
            "highest": market_data.get("highest"),
            "last_btc": market_data.get("last_btc")
        }

        updated = False

        for coin in data["coins"]:
            if not isinstance(coin, dict):
                continue

            if (
                coin.get("coin")
                and normalize(coin.get("symbol")) == normalize(symbol)
            ):
                old_cache = coin.get("cached_market_data", {})

                if not isinstance(old_cache, dict):
                    old_cache = {}

                coin["cached_market_data"] = {
                    **old_cache,
                    **new_cache
                }

                updated = True
                break

        # For coins not present in static records,
        # retain one market snapshot.
        if not updated:
            new_record = {
                "symbol": symbol,
                **market_data
            }

            data["coins"].append(new_record)

        # Remove old API-only duplicates for this symbol.
        # Static records are never removed.
        matching_snapshots = [
            coin for coin in get_snapshot_records(data)
            if normalize(coin.get("symbol")) == normalize(symbol)
        ]

        if updated:
            data["coins"] = [
                coin for coin in data["coins"]
                if not (
                    isinstance(coin, dict)
                    and not coin.get("coin")
                    and normalize(coin.get("symbol"))
                    == normalize(symbol)
                )
            ]

        elif len(matching_snapshots) > 1:
            latest = get_latest_snapshot(symbol, data)

            data["coins"] = [
                coin for coin in data["coins"]
                if not (
                    isinstance(coin, dict)
                    and not coin.get("coin")
                    and normalize(coin.get("symbol"))
                    == normalize(symbol)
                )
            ]

            data["coins"].append(latest)

        data.setdefault("metadata", {})
        data["metadata"]["last_updated"] = (
            datetime.now(timezone.utc).isoformat()
        )

        save_kb(data)


@tool
def kb(query: str) -> str:
    """Search the local cryptocurrency knowledge base.

    Input must be a cryptocurrency name or ticker,
    not the entire user message.

    Returns static facts and historical cached data.
    Cached market data must not be described as live.
    """

    try:
        data = load_kb()

        coin = find_static_coin(query, data)

        if coin:
            return json.dumps({
                "status": "success",
                "source": "Knowledge Base",
                "coin": coin,
                "market_data_is_live": False
            })

        resolved = resolve_coin(query)

        if resolved:
            snapshot = get_latest_snapshot(
                resolved["symbol"],
                data
            )

            if snapshot:
                return json.dumps({
                    "status": "success",
                    "source": "Knowledge Base",
                    "coin": {
                        "coin": resolved["name"],
                        "symbol": resolved["symbol"],
                        "cached_market_data": {
                            "last_price": snapshot.get("last"),
                            "price_timestamp": snapshot.get("date"),
                            "daily_change_percentage": snapshot.get(
                                "daily_change_percentage"
                            ),
                            "source_exchange": snapshot.get(
                                "source_exchange"
                            )
                        }
                    },
                    "market_data_is_live": False
                })

        return json.dumps({
            "status": "not_found",
            "query": query
        })

    except Exception as e:
        logger.exception("KB lookup failed")

        return json.dumps({
            "status": "error",
            "error": str(e)
        })


@tool
def get_crypto_price(query: str) -> str:
    """Fetch fresh cryptocurrency market data.

    Input must be one cryptocurrency name or ticker.

    Use for current prices and supported live market
    statistics. Do not use cached data as a fallback
    when the live API fails.
    """

    try:
        if not API_KEY:
            return json.dumps({
                "status": "error",
                "error": "CRYPTO_API_KEY is missing"
            })

        resolved = resolve_coin(query)

        if not resolved:
            return json.dumps({
                "status": "error",
                "error": (
                    "Could not uniquely identify the "
                    "requested cryptocurrency"
                )
            })

        symbol = resolved["symbol"]

        response = requests.get(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}"
            },
            params={
                "symbol": symbol
            },
            timeout=15
        )

        response.raise_for_status()

        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError("Invalid API response structure")

        symbols = payload.get("symbols")

        if not isinstance(symbols, list):
            return json.dumps({
                "status": "error",
                "error": "API did not return a symbols list"
            })

        matching = [
            item for item in symbols
            if (
                isinstance(item, dict)
                and normalize(item.get("symbol"))
                == normalize(symbol)
                and item.get("last") is not None
            )
        ]

        if not matching:
            return json.dumps({
                "status": "not_found",
                "symbol": symbol,
                "error": "No matching market data returned"
            })

        # Select the most recent matching quote.
        matching.sort(
            key=lambda item: (
                parse_timestamp(item.get("date"))
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
            reverse=True
        )

        market_data = matching[0]

        quote_time = parse_timestamp(market_data.get("date"))

        if quote_time is None:
            return json.dumps({
                "status": "error",
                "error": "API quote has no valid timestamp"
            })

        quote_age = (
            datetime.now(timezone.utc) - quote_time
        ).total_seconds()

        if (
            quote_age < -300
            or quote_age > MAX_QUOTE_AGE_SECONDS
        ):
            return json.dumps({
                "status": "error",
                "error": "API quote is stale or has an invalid time",
                "quote_timestamp": market_data.get("date")
            })

        try:
            price = float(market_data["last"])

            if price <= 0:
                raise ValueError("Invalid price")

        except (TypeError, ValueError):
            return json.dumps({
                "status": "error",
                "error": "API returned an invalid price"
            })

        # Only update the KB after validation.
        kb_updated = False

        try:
            update_market_cache(symbol, market_data)
            kb_updated = True

        except Exception:
            logger.exception("Could not update KB")

        return json.dumps({
            "status": "success",
            "source": "External Tool",
            "coin": resolved["name"],
            "symbol": symbol,
            "market_data": market_data,
            "currency": "USD",
            "market_data_is_live": True,
            "kb_updated": kb_updated
        })

    except requests.exceptions.RequestException as e:
        logger.exception("Crypto API request failed")

        return json.dumps({
            "status": "error",
            "error": f"API request failed: {e}"
        })

    except Exception as e:
        logger.exception("Crypto lookup failed")

        return json.dumps({
            "status": "error",
            "error": str(e)
        })
