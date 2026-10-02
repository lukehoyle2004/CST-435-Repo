"""Kept for the tutorial's ``python -m db.seed`` command.

Income-Insight now trains on the real UCI Adult data, so seeding means loading
the ``adult_income`` table. See :mod:`db.load_adult`.
"""
from db.load_adult import main

if __name__ == "__main__":
    main()
