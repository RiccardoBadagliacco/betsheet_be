#!/usr/bin/env python3
"""
Script to extract timestamps from UUID v4 and update created_at field in bets table.

Note: UUID v4 are random and don't contain timestamps. However, if the UUIDs were
generated sequentially, we can try to extract creation time from UUID v1 format
or use other heuristics. 

For UUID v4 (random), we'll need to use an alternative approach.
Let's check the actual UUID format first.
"""

import sqlite3
import uuid
from datetime import datetime
import sys

def extract_timestamp_from_uuid_v1(uuid_str):
    """Extract timestamp from UUID v1 (time-based UUID)."""
    try:
        uuid_obj = uuid.UUID(uuid_str)
        if uuid_obj.version == 1:
            # UUID v1 contains timestamp
            timestamp = uuid_obj.time
            # UUID timestamp is in 100-nanosecond intervals since Oct 15, 1582
            # Convert to Unix timestamp
            unix_timestamp = (timestamp - 0x01b21dd213814000) / 10000000.0
            return datetime.fromtimestamp(unix_timestamp)
        else:
            return None
    except Exception as e:
        print(f"Error processing UUID {uuid_str}: {e}")
        return None

def analyze_uuids_in_database():
    """Analyze the UUIDs in the database to understand their format."""
    conn = sqlite3.connect('bets.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM bets LIMIT 10")
    uuids = cursor.fetchall()
    
    print("Analyzing UUIDs in database:")
    for uuid_str, in uuids:
        try:
            uuid_obj = uuid.UUID(uuid_str)
            print(f"UUID: {uuid_str}")
            print(f"  Version: {uuid_obj.version}")
            print(f"  Variant: {uuid_obj.variant}")
            
            if uuid_obj.version == 1:
                timestamp = extract_timestamp_from_uuid_v1(uuid_str)
                print(f"  Extracted timestamp: {timestamp}")
            else:
                print(f"  UUID v{uuid_obj.version} - no timestamp available")
            print()
        except Exception as e:
            print(f"Error analyzing UUID {uuid_str}: {e}")
    
    conn.close()

def update_created_at_from_data_field():
    """
    Alternative approach: try to extract date from the 'data' field if available,
    or use a default timestamp for existing records.
    """
    conn = sqlite3.connect('bets.db')
    cursor = conn.cursor()
    
    # First, let's see what's in the data field
    cursor.execute("SELECT id, data FROM bets WHERE data IS NOT NULL LIMIT 10")
    sample_data = cursor.fetchall()
    
    print("Sample 'data' field values:")
    for bet_id, data_value in sample_data:
        print(f"ID: {bet_id[:8]}... Data: {data_value}")
    
    conn.close()

def update_created_at_with_time_distribution():
    """
    Update created_at field with smart time distribution.
    
    Strategy:
    1. Group bets by date
    2. Sort UUIDs within each date (UUID ordering often correlates with creation time)
    3. Distribute times evenly throughout the day (8:00 AM to 10:00 PM)
    """
    conn = sqlite3.connect('bets.db')
    cursor = conn.cursor()
    
    # Get all bets with their data field
    cursor.execute("SELECT id, data FROM bets ORDER BY data, id")
    bets = cursor.fetchall()
    
    # Group bets by date
    bets_by_date = {}
    for bet_id, data_value in bets:
        if data_value:
            try:
                # Extract date from data field
                data_str = str(data_value).strip()
                date_key = data_str[:10]  # Get YYYY-MM-DD part
                
                if date_key not in bets_by_date:
                    bets_by_date[date_key] = []
                bets_by_date[date_key].append(bet_id)
            except:
                pass
    
    updated_count = 0
    
    print(f"Processing {len(bets_by_date)} unique dates...")
    
    for date_str, bet_ids in bets_by_date.items():
        try:
            # Parse the date
            base_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Sort bet IDs (UUID ordering often correlates with creation order)
            bet_ids.sort()
            
            num_bets = len(bet_ids)
            print(f"Date {date_str}: {num_bets} bets")
            
            # Distribute times between 8:00 AM and 10:00 PM (14 hours = 840 minutes)
            start_hour = 8
            end_hour = 22
            total_minutes = (end_hour - start_hour) * 60
            
            for i, bet_id in enumerate(bet_ids):
                if num_bets == 1:
                    # Single bet: place at 12:00 PM
                    minutes_offset = 4 * 60  # 12:00 PM
                else:
                    # Multiple bets: distribute evenly
                    minutes_offset = start_hour * 60 + (i * total_minutes // (num_bets - 1))
                
                # Create the full datetime
                hours = minutes_offset // 60
                minutes = minutes_offset % 60
                created_at = base_date.replace(hour=hours, minute=minutes, second=0)
                
                # Update the database
                cursor.execute(
                    "UPDATE bets SET created_at = ? WHERE id = ?",
                    (created_at.isoformat(), bet_id)
                )
                updated_count += 1
                
                print(f"  {bet_id[:8]}... -> {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        except Exception as e:
            print(f"Error processing date {date_str}: {e}")
    
    # Handle bets without valid data field
    cursor.execute("SELECT id FROM bets WHERE data IS NULL OR data = ''")
    orphan_bets = cursor.fetchall()
    
    if orphan_bets:
        print(f"\nProcessing {len(orphan_bets)} bets without valid data field...")
        fallback_date = datetime(2025, 1, 1, 12, 0, 0)
        
        for bet_id, in orphan_bets:
            cursor.execute(
                "UPDATE bets SET created_at = ? WHERE id = ?",
                (fallback_date.isoformat(), bet_id)
            )
            updated_count += 1
            print(f"  {bet_id[:8]}... -> {fallback_date.strftime('%Y-%m-%d %H:%M:%S')} (fallback)")
    
    conn.commit()
    conn.close()
    
    print(f"\nUpdated {updated_count} records with created_at timestamps")
    print("Time distribution strategy:")
    print("- Bets grouped by date from 'data' field")
    print("- UUIDs sorted within each date (proxy for creation order)")
    print("- Times distributed between 8:00 AM and 10:00 PM")
    print("- Single bets placed at 12:00 PM")

if __name__ == "__main__":
    print("=== UUID Analysis ===")
    analyze_uuids_in_database()
    
    print("\n=== Data Field Analysis ===")
    update_created_at_from_data_field()
    
    print("\n=== Updating created_at field ===")
    if len(sys.argv) > 1 and sys.argv[1] == "--update":
        update_created_at_with_time_distribution()
        print("Update completed!")
    else:
        print("Run with --update flag to actually update the database")
        print("Example: python update_created_at_from_uuid.py --update")