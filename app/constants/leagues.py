"""
Constants for football leagues supported by football-data.co.uk
Structure: {league_code: {"name": league_name, "country": country_name}}
"""

LEAGUES = {
    # Major European Leagues - Tier 1
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

# Top tier leagues for quick access
TOP_TIER_LEAGUES = ["E0", "D1", "I1", "SP1", "F1", "N1", "P1"]

def get_league_info(league_code: str) -> dict:
    """
    Get league information by code.
    
    Args:
        league_code: The league code (e.g., 'I1')
        
    Returns:
        Dict with name, country, or empty dict if not found
    """
    return LEAGUES.get(league_code.upper(), {})

def get_leagues_by_country(country: str) -> list:
    """
    Get all league codes for a specific country.
    
    Args:
        country: Country name (e.g., 'Italy')
        
    Returns:
        List of league codes for that country
    """
    return COUNTRIES.get(country, [])

def is_top_tier_league(league_code: str) -> bool:
    """
    Check if a league is considered top tier.
    
    Args:
        league_code: The league code (e.g., 'I1')
        
    Returns:
        True if top tier, False otherwise
    """
    return league_code.upper() in TOP_TIER_LEAGUES

def get_all_countries() -> list:
    """
    Get list of all supported countries.
    
    Returns:
        List of country names
    """
    return list(COUNTRIES.keys())

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