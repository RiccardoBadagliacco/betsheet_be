from typing import Dict, Any

def format_stats_for_side(stats: Dict[str, Any], side: str) -> Dict[str, Any]:
    """
    Converte le statistiche grezze di get_team_stats in una struttura coerente
    con il formato richiesto dal front-end e dai calcoli dei picchetti tecnici.
    """

    formatted = {}

    for market, data in stats.items():
        # market può essere: "1X2", "OU15", "OU25", "GNG"
        formatted_market = {"type": market}

        # per ogni sezione (totali, recenti, home/away, ecc.)
        for section_name, section_data in data.items():
            # es: section_name = "totali", section_data = {"partite": 10, "vittorie": 5, ...}
            formatted_market[section_name] = {
                "label": label_from_section(section_name, side),
                "stats": {
                    key: {"value": val, "label": short_label(key)}
                    for key, val in section_data.items()
                }
            }

        formatted[market] = formatted_market

    return formatted


def label_from_section(section_name: str, side: str) -> str:
    """Restituisce una label leggibile per la sezione."""
    if section_name == "totali":
        return "Totale stagione"
    if section_name == "recenti":
        return "Ultime 5 partite"
    if "home" in section_name.lower() or (side == "home" and "side" in section_name):
        return "Totale HOME" if "totali" in section_name else "Ultime 5 partite HOME"
    if "away" in section_name.lower() or (side == "away" and "side" in section_name):
        return "Totale AWAY" if "totali" in section_name else "Ultime 5 partite AWAY"
    return section_name.capitalize()


def short_label(key: str) -> str:
    """Ritorna etichette corte per le statistiche."""
    mapping = {
        "partite": "N",
        "vittorie": "V",
        "pareggi": "P",
        "sconfitte": "S",
        "under": "U",
        "over": "O",
        "goal": "G",
        "no_goal": "NG",
    }
    return mapping.get(key, key[:2].upper())