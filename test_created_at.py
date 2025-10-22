#!/usr/bin/env python3
"""
Test script to verify that created_at field is automatically populated when creating a new bet.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.database import SessionLocal
from app.db.models import Bet
from app.schemas.bet import BetCreate
from app.api.bets import create_bet
from datetime import datetime
import uuid

def test_created_at_auto_population():
    """Test that created_at is automatically populated when creating a bet."""
    
    # Create a session
    db = SessionLocal()
    
    try:
        # Create a test bet payload
        test_bet = BetCreate(
            importo=10.0,
            quota=2.0,
            esito="vinta",
            data="2025-10-22",
            backroll_id="0ed2596a-8203-4426-96ca-ba672628ca82",  # Scalata backroll
            note="Test bet for created_at verification"
        )
        
        print("Creating test bet...")
        print(f"Test bet data: {test_bet.dict()}")
        
        # Call the create_bet function
        result = create_bet(test_bet, db)
        
        print(f"\nBet created successfully!")
        print(f"ID: {result['id']}")
        print(f"Data: {result['data']}")
        print(f"Created at: {result.get('created_at', 'NOT FOUND!')}")
        print(f"Importo: {result['importo']}")
        print(f"Quota: {result['quota']}")
        print(f"Esito: {result['esito']}")
        print(f"Vincita: {result['vincita']}")
        print(f"Profitto: {result['profitto']}")
        
        # Verify in database
        bet_in_db = db.query(Bet).filter(Bet.id == result['id']).first()
        if bet_in_db:
            print(f"\nVerification from database:")
            print(f"DB created_at: {bet_in_db.created_at}")
            print(f"DB created_at type: {type(bet_in_db.created_at)}")
            
            if bet_in_db.created_at:
                print("✅ SUCCESS: created_at field is populated in database!")
                
                # Check if it's recent (within last minute)
                now = datetime.utcnow()
                time_diff = abs((now - bet_in_db.created_at).total_seconds())
                print(f"Time difference from now: {time_diff:.2f} seconds")
                
                if time_diff < 60:  # Less than 1 minute
                    print("✅ SUCCESS: created_at timestamp is recent and correct!")
                else:
                    print("❌ WARNING: created_at timestamp seems old")
            else:
                print("❌ FAILED: created_at field is NULL in database!")
        else:
            print("❌ FAILED: Bet not found in database!")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_created_at_auto_population()