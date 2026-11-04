#!/usr/bin/env python3
"""
Database initialization script
Run this script to create all database tables
"""

from .models.database import init_db

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Database initialization complete!")
