"""
Constants for football leagues supported by football-data.co.uk
Structure: {league_code: {"name": league_name, "country": country_name}}
"""


""" 
LEAGUES = {
    # Major European Leagues - Tier 1
    "main": {
        "E0": {
        "name": "Premier League",
        "country": "England"
        },
        "E1": {
            "name": "Championship", 
            "country": "England"
        },
        
        "SC0": {
            "name": "Premier League",
            "country": "Scotland"
        },
        
        "D1": {
            "name": "Bundesliga",
            "country": "Germany"
        },
        "D2": {
            "name": "2. Bundesliga",
            "country": "Germany"
        },
        
        "I1": {
            "name": "Serie A",
            "country": "Italy"
        },
        "I2": {
            "name": "Serie B",
            "country": "Italy"
        },
    
        "SP1": {
            "name": "La Liga",
            "country": "Spain"
        },
        "SP2": {
            "name": "Segunda División",
            "country": "Spain"
        },
        
        "F1": {
            "name": "Ligue 1",
            "country": "France"
        },
        "F2": {
            "name": "Ligue 2",
            "country": "France"
        },
        
        "N1": {
            "name": "Eredivisie",
            "country": "Netherlands"
        },
        
        "B1": {
            "name": "Jupiler Pro League",
            "country": "Belgium"
        },
    
        "P1": {
            "name": "Primeira Liga",
            "country": "Portugal"
        },
        
        "T1": {
            "name": "Süper Lig",
            "country": "Turkey"
        },
    },
    "other": {
        "AUT": {
            "name": "Austrian Bundesliga",
            "country": "Austria"
        },
        "CHN": {
            "name": "Chinese Super League",
            "country": "China"
        },
    }   
}

# Countries grouped for easy reference
COUNTRIES = {
    "England": ["E0", "E1", "E2", "E3", "EC"],
    "Scotland": ["SC0", "SC1", "SC2", "SC3"],
    "Germany": ["D1", "D2"],
    "Italy": ["I1", "I2"],
    "Spain": ["SP1", "SP2"],
    "France": ["F1", "F2"],
    "Netherlands": ["N1"],
    "Belgium": ["B1"],
    "Portugal": ["P1"],
    "Turkey": ["T1"],
    "Greece": ["G1"],
    "Argentina": ["ARG"],
    "Brazil": ["BRA"],
    "Mexico": ["MEX"],
    "USA": ["USA"],
    "China": ["CHN"],
    "Japan": ["JPN"],
}
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
        {"league": {"main": False, "name": "Austrian Bundesliga", "code": "AUT"}}
    ],
    "China": [
        {"league": {"main": False, "name": "Chinese Super League", "code": "CHN"}}
    ]
}