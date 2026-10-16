"""Aggregate each traffic table at its own grain; never multiply people by vehicles."""
import pandas as pd


def summarize_crashes(crashes, vehicles, people):
    required = [(crashes, ['CRASH_RECORD_ID', 'INJURIES_TOTAL']), (vehicles, ['CRASH_RECORD_ID', 'CRASH_UNIT_ID', 'MAKE']), (people, ['CRASH_RECORD_ID', 'PERSON_ID', 'AGE'])]
    for frame, columns in required:
        if not set(columns).issubset(frame):
            raise ValueError(f'Missing required columns: {columns}')
    for frame, key in [(crashes, 'CRASH_RECORD_ID'), (vehicles, 'CRASH_UNIT_ID'), (people, 'PERSON_ID')]:
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f'{key} must be unique and non-null at its own grain')
    crash_ids = set(crashes.CRASH_RECORD_ID)
    if not set(vehicles.CRASH_RECORD_ID).issubset(crash_ids) or not set(people.CRASH_RECORD_ID).issubset(crash_ids):
        raise ValueError('Child records refer to missing crashes')
    injuries = pd.to_numeric(crashes.INJURIES_TOTAL, errors='coerce')
    if (injuries.dropna() < 0).any():
        raise ValueError('Injury counts cannot be negative')
    exposure = vehicles.assign(MAKE=vehicles.MAKE.fillna('UNKNOWN')).groupby('MAKE').agg(vehicle_records=('CRASH_UNIT_ID', 'count'), unique_crashes=('CRASH_RECORD_ID', 'nunique')).reset_index()
    ages = pd.to_numeric(people.AGE, errors='coerce')
    valid_age = ages.between(0, 120)
    return {'crashCount': len(crashes), 'vehicleCount': len(vehicles), 'personCount': len(people),
            'knownInjuries': int(injuries.sum()), 'missingInjuryCounts': int(injuries.isna().sum()),
            'makes': exposure.sort_values('vehicle_records', ascending=False),
            'ages': ages[valid_age].copy(), 'excludedAges': int((~valid_age).sum())}
