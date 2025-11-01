"""
Constants for football leagues supported by football-data.co.uk
Structure: {league_code: {"name": league_name, "country": country_name}}
"""


# Top tier  for quick access
TOP_TIER_LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "N1", "P1"]

def get_league_info(league_code: str) -> dict:
    """
    Get league information by code.

    Args:
        league_code: The league code (e.g., 'I1')

    Returns:
        Dict with name and country, or empty dict if not found.
    """
    for country, leagues in COUNTRY_LEAGUE_STRUCTURE.items():
        for league_entry in leagues:
            league = league_entry["league"]
            if league["code"].upper() == league_code.upper():
                return {"name": league["name"], "country": country}
    return {}

def get_leagues_by_country(country: str) -> list:
    """
    Get all league codes for a specific country.

    Args:
        country: Country name (e.g., 'Italy')

    Returns:
        List of league codes for that country.
    """
    if country in COUNTRY_LEAGUE_STRUCTURE:
        return [league_entry["league"]["code"] for league_entry in COUNTRY_LEAGUE_STRUCTURE[country]]
    return []



def get_all_countries() -> list:
    """
    Get list of all supported countries.

    Returns:
        List of country names
    """
    return list(COUNTRY_LEAGUE_STRUCTURE.keys())

def get_formatted_league_name(league_code: str) -> str:
    """
    Get formatted league name with country.

    Args:
        league_code: The league code (e.g., 'I1')

    Returns:
        Formatted string like "Serie A (Italy)" or "Unknown League (CODE)"
    """
    info = get_league_info(league_code)
    if info:
        return f"{info['name']} ({info['country']})"
    else:
        return f"Unknown League ({league_code.upper()})"
    
    
def get_all_leagues(type='main') -> dict:
    """
    Get all leagues information.

    Args:
        type: 'main', 'other', or 'all'.

    Returns:
        Dict of leagues in the format:
        {
            "leagueCode": {"name": "League Name", "country": "Country Name"}
        }
    """
    result = {}
    for country, leagues in COUNTRY_LEAGUE_STRUCTURE.items():
        for league_entry in leagues:
            league = league_entry["league"]
            if type == 'all' or (type == 'main' and league["main"]) or (type == 'other' and not league["main"]):
                result[league["code"]] = {"name": league["name"], "country": country}
    return result


# Oggetto statico COUNTRY_LEAGUE_STRUCTURE popolato manualmente
COUNTRY_LEAGUE_STRUCTURE = {
    "England": [
        {"league": {"main": True, "name": "Premier League", "code": "E0"}},
        {"league": {"main": True, "name": "Championship", "code": "E1"}}
    ],
    "Scotland": [
        {"league": {"main": True, "name": "Premier League", "code": "SC0"}}
    ],
    "Germany": [
        {"league": {"main": True, "name": "Bundesliga", "code": "D1"}},
        {"league": {"main": True, "name": "2. Bundesliga", "code": "D2"}}
    ],
    "Italy": [
        {"league": {"main": True, "name": "Serie A", "code": "I1"}},
        {"league": {"main": True, "name": "Serie B", "code": "I2"}}
    ],
    "Spain": [
        {"league": {"main": True, "name": "La Liga", "code": "SP1"}},
        {"league": {"main": True, "name": "Segunda División", "code": "SP2"}}
    ],
    "France": [
        {"league": {"main": True, "name": "Ligue 1", "code": "F1"}},
        {"league": {"main": True, "name": "Ligue 2", "code": "F2"}}
    ],
    "Netherlands": [
        {"league": {"main": True, "name": "Eredivisie", "code": "N1"}}
    ],
    "Belgium": [
        {"league": {"main": True, "name": "Jupiler Pro League", "code": "B1"}}
    ],
    "Portugal": [
        {"league": {"main": True, "name": "Primeira Liga", "code": "P1"}}
    ],
    "Turkey": [
        {"league": {"main": True, "name": "Süper Lig", "code": "T1"}}
    ],
    "Austria": [
        {"league": {"main": False, "name": "Bundesliga", "code": "AUT"}}
    ],
    "China": [
        {"league": {"main": False, "name": "Super League", "code": "CHN"}}
    ],
    "Denmark": [
        {"league": {"main": False, "name": "Superliga", "code": "DNK"}}
    ],
    "Finland": [
        {"league": {"main": False, "name": "Veikkausliiga", "code": "FIN"}}
    ],
    "Ireland": [
        {"league": {"main": False, "name": "Premier Division", "code": "IRL"}}
    ],
    "Norway": [
        {"league": {"main": False, "name": "Eliteserien", "code": "NOR"}}
    ],
    "Sweden": [
        {"league": {"main": False, "name": "Allsvenskan", "code": "SWE"}}
    ]
}

def get_league_name_from_code(league_code: str) -> str:
    """
    Converte il codice lega nel nome completo

    Args:
        league_code: Codice della lega (es. "E0", "I1")

    Returns:
        Nome completo della lega
    """
    for country, leagues in COUNTRY_LEAGUE_STRUCTURE.items():
        for league_entry in leagues:
            league = league_entry["league"]
            if league["code"].upper() == league_code.upper():
                return league["name"]
    return f"League {league_code}"


def get_country_from_league_code(league_code: str) -> str:
    """
    Mappa il codice lega al paese corrispondente

    Args:
        league_code: Codice della lega (es. "E0", "I1")

    Returns:
        Nome del paese corrispondente o 'Unknown' se non trovato
    """
    for country, leagues in COUNTRY_LEAGUE_STRUCTURE.items():
        for league_entry in leagues:
            league = league_entry["league"]
            if league["code"].upper() == league_code.upper():
                return country
    return 'Unknown'