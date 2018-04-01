#!/usr/bin/env python
# coding: utf-8


"""
Usage:
        ./fixed-invest.py
"""

import os.path


def parse_csv(csv_path):
    rows = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:-2]:
            cells = line.split(',')
            row = {
                "date": cells[0],
                "price": float(cells[1]),
                "cap": float(cells[2]),
                'pe-ttm': float(cells[3])
            }
            rows.append(row)
    return rows


def main():
    csv_path = os.path.dirname(os.path.abspath(__file__)) + '/hs300_pe_ttm_20180401_171619.csv'
    print parse_csv(csv_path)


if __name__ == '__main__':
    main()
