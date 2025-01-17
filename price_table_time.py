import pandas as pd

# price table and summer period

off_peak_days = [
    '2024-01-01',
    '2024-02-09',
    '2024-02-10',
    '2024-02-11',
    '2024-02-12',
    '2024-02-13',
    '2024-02-14',
    '2024-02-28',
    '2024-04-04',
    '2024-05-01',
    '2024-06-10',
    '2024-09-17',
    '2024-10-10'
]

# basic price
basic_price = {
    'summer': 217.30,
    'normal': 160.60
}

# high voltage two-section type pricing table
price_section_2 = {
    'summer_peak':5.32,
    'summer_sat':2.40,
    'summer_offpeak':2.20,
    'normal_peak':4.99,
    'normal_sat':2.18,
    'normal_offpeak':1.97
}

# high voltage three-section type pricing table
price_section_3 = {
    'summer_highpeak':7.49,
    'summer_peak':4.64,
    'summer_sat':2.20,
    'summer_offpeak':2.08,
    'normal_peak':4.34,
    'normal_sat':2.03,
    'normal_offpeak':1.89
}

# setting the summer period
summer_start = pd.to_datetime('2024-05-16')
summer_end = pd.to_datetime('2024-10-15')
