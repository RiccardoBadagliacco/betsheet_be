#!/usr/bin/env python3
"""
Confronto Performance Serie A vs Premier League 2025/2026
"""

import pandas as pd
import numpy as np

def main():
    print("⚽ CONFRONTO SERIE A vs PREMIER LEAGUE 2025/2026")
    print("=" * 60)
    
    try:
        # Carica dati Serie A
        print("📊 Caricando dati Serie A...")
        seria_df = pd.read_csv('reports/backtest_complete_2025_2026.csv')
        
        # Carica dati Premier League
        print("📊 Caricando dati Premier League...")
        pl_df = pd.read_csv('reports/backtest_premier_league_2025_2026.csv')
        
        print(f"\n📈 CONFRONTO PERFORMANCE:")
        print("-" * 40)
        
        # Over 0.5
        seria_o05 = seria_df['actual_O_0_5'].mean() if 'actual_O_0_5' in seria_df.columns else 0
        pl_o05 = pl_df['actual_O_0_5'].mean() if 'actual_O_0_5' in pl_df.columns else 0
        
        print(f"Over 0.5 Goals:")
        print(f"  🇮🇹 Serie A:        {seria_o05:.1%} ({len(seria_df)} partite)")
        print(f"  🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League:  {pl_o05:.1%} ({len(pl_df)} partite)")
        print(f"  📊 Differenza:      {(pl_o05 - seria_o05)*100:+.1f}%")
        
        # Over 1.5
        seria_o15 = seria_df['actual_O_1_5'].mean() if 'actual_O_1_5' in seria_df.columns else 0
        pl_o15 = pl_df['actual_O_1_5'].mean() if 'actual_O_1_5' in pl_df.columns else 0
        
        print(f"\nOver 1.5 Goals:")
        print(f"  🇮🇹 Serie A:        {seria_o15:.1%} ({len(seria_df)} partite)")
        print(f"  🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League:  {pl_o15:.1%} ({len(pl_df)} partite)")
        print(f"  📊 Differenza:      {(pl_o15 - seria_o15)*100:+.1f}%")
        
        # Multigol Casa 1-3
        seria_mg_casa = seria_df['actual_MG_Casa_1_3'].mean() if 'actual_MG_Casa_1_3' in seria_df.columns else 0
        pl_mg_casa = pl_df['actual_MG_Casa_1_3'].mean() if 'actual_MG_Casa_1_3' in pl_df.columns else 0
        
        print(f"\nMultigol Casa 1-3:")
        print(f"  🇮🇹 Serie A:        {seria_mg_casa:.1%}")
        print(f"  🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League:  {pl_mg_casa:.1%}")
        print(f"  📊 Differenza:      {(pl_mg_casa - seria_mg_casa)*100:+.1f}%")
        
        # Multigol Ospite 1-3
        seria_mg_ospite = seria_df['actual_MG_Ospite_1_3'].mean() if 'actual_MG_Ospite_1_3' in seria_df.columns else 0
        pl_mg_ospite = pl_df['actual_MG_Ospite_1_3'].mean() if 'actual_MG_Ospite_1_3' in pl_df.columns else 0
        
        print(f"\nMultigol Ospite 1-3:")
        print(f"  🇮🇹 Serie A:        {seria_mg_ospite:.1%}")
        print(f"  🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League:  {pl_mg_ospite:.1%}")
        print(f"  📊 Differenza:      {(pl_mg_ospite - seria_mg_ospite)*100:+.1f}%")
        
        print(f"\n🎯 ANALISI COMPARATIVA:")
        print("-" * 40)
        
        if pl_o05 > seria_o05:
            print("• Premier League è più offensiva (più Over 0.5)")
        else:
            print("• Serie A è più offensiva (più Over 0.5)")
            
        if pl_o15 > seria_o15:
            print("• Premier League produce più gol totali (Over 1.5)")
        else:
            print("• Serie A produce più gol totali (Over 1.5)")
            
        if pl_mg_casa > seria_mg_casa:
            print("• Squadre di casa più forti in Premier League")
        else:
            print("• Squadre di casa più forti in Serie A")
        
        print(f"\n📋 RACCOMANDAZIONI STRATEGICHE:")
        print("-" * 40)
        print("🇮🇹 Serie A - Migliori mercati:")
        if seria_o05 > 0.85:
            print("  ✅ Over 0.5 Goals (alta affidabilità)")
        if seria_mg_casa > 0.7:
            print("  ✅ Multigol Casa 1-3 (buona performance)")
            
        print("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League - Migliori mercati:")  
        if pl_o05 > 0.9:
            print("  ✅ Over 0.5 Goals (eccellente performance)")
        if pl_o15 > 0.75:
            print("  ✅ Over 1.5 Goals (alta produttività)")
        if pl_mg_casa > 0.72:
            print("  ✅ Multigol Casa (squadre di casa forti)")
            
    except FileNotFoundError as e:
        print(f"❌ Errore: File non trovato - {e}")
        print("   Assicurati di aver eseguito i backtest per entrambi i campionati.")
    except Exception as e:
        print(f"❌ Errore durante l'analisi: {e}")

if __name__ == "__main__":
    main()